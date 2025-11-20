import os
import uuid
import json
import threading
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Text, DateTime, String
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

from src.planning_agent import planner_agent, executor_agent_step

import html

# === Load env vars ===
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./research_agent.db")

# Fix for Heroku's postgres:// URL format
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# === DB setup ===
Base = declarative_base()
engine = create_engine(
    DATABASE_URL, 
    echo=False, 
    future=True,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(bind=engine)


class Task(Base):
    __tablename__ = "tasks"
    id = Column(String, primary_key=True, index=True)
    prompt = Column(Text)
    status = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    result = Column(Text)


# Drop and recreate tables (for development)
try:
    Base.metadata.drop_all(bind=engine)
    print(" Dropped existing tables")
except Exception as e:
    print(f" Table drop warning: {e}")

try:
    Base.metadata.create_all(bind=engine)
    print(" Database tables created successfully")
except Exception as e:
    print(f" DB creation failed: {e}")
    raise

# === FastAPI ===
app = FastAPI(title="AI Research Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Mount static files and templates
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    print(" Static files mounted")
except Exception as e:
    print(f" Static files not found: {e}")

templates = Jinja2Templates(directory="templates")

# In-memory progress tracking
task_progress = {}


class PromptRequest(BaseModel):
    prompt: str


@app.get("/", response_class=HTMLResponse)
def read_index(request: Request):
    """Serve the main HTML interface"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api", response_class=JSONResponse)
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/generate_report")
def generate_report(req: PromptRequest):
    """
    Endpoint to initiate report generation
    
    1. Creates task in DB
    2. Generates research plan
    3. Spawns background thread for execution
    4. Returns task_id for progress tracking
    """
    task_id = str(uuid.uuid4())
    
    # Create task in database
    db = SessionLocal()
    try:
        db.add(Task(
            id=task_id, 
            prompt=req.prompt, 
            status="running",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        ))
        db.commit()
        print(f" Task {task_id} created in database")
    except Exception as e:
        print(f" Database error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

    # Initialize progress tracking
    task_progress[task_id] = {"steps": []}
    
    # Generate research plan
    try:
        print(f" Generating plan for: {req.prompt[:100]}...")
        initial_plan_steps = planner_agent(req.prompt)
        print(f" Plan generated: {len(initial_plan_steps)} steps")
        
        # Populate progress tracker with pending steps
        for step_title in initial_plan_steps:
            task_progress[task_id]["steps"].append({
                "title": step_title,
                "status": "pending",
                "description": "Awaiting execution",
                "substeps": [],
                "updated_at": datetime.utcnow().isoformat()
            })
    except Exception as e:
        print(f" Planning error: {e}")
        # Update task status to error
        db = SessionLocal()
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "error"
            task.updated_at = datetime.utcnow()
            db.commit()
        db.close()
        raise HTTPException(status_code=500, detail=f"Planning failed: {str(e)}")

    # Start background execution thread
    thread = threading.Thread(
        target=run_agent_workflow, 
        args=(task_id, req.prompt, initial_plan_steps),
        daemon=True
    )
    thread.start()
    print(f" Background thread started for task {task_id}")
    
    return {"task_id": task_id, "steps_count": len(initial_plan_steps)}


@app.get("/task_progress/{task_id}")
def get_task_progress(task_id: str):
    """
    Endpoint for real-time progress updates
    Frontend polls this to update UI
    """
    if task_id not in task_progress:
        raise HTTPException(status_code=404, detail="Task not found in progress tracker")
    
    return task_progress.get(task_id, {"steps": []})


@app.get("/task_status/{task_id}")
def get_task_status(task_id: str):
    """
    Endpoint to retrieve final task status and results from database
    """
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found in database")
        
        result_data = None
        if task.result:
            try:
                result_data = json.loads(task.result)
            except json.JSONDecodeError:
                result_data = {"raw": task.result}
        
        return {
            "task_id": task_id,
            "status": task.status,
            "prompt": task.prompt,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
            "result": result_data
        }
    finally:
        db.close()


def format_history(history: list) -> str:
    """Format execution history for display"""
    formatted = []
    for title, desc, output in history:
        formatted.append(f"📹 {title}\n{desc}\n\n📄 Output:\n{output}")
    return "\n\n".join(formatted)


def run_agent_workflow(task_id: str, prompt: str, initial_plan_steps: list):
    """
    Background workflow executor
    
    Executes each step in the plan sequentially:
    1. Updates step status to "running"
    2. Calls appropriate agent via executor_agent_step()
    3. Appends output to execution history
    4. Updates step status to "done" with formatted results
    5. Stores final report in database
    """
    steps_data = task_progress[task_id]["steps"]
    execution_history = []

    def update_step_status(index: int, status: str, description: str = "", substep: dict = None):
        """Helper to update step status in progress tracker"""
        if index < len(steps_data):
            steps_data[index]["status"] = status
            if description:
                steps_data[index]["description"] = description
            if substep:
                steps_data[index]["substeps"].append(substep)
            steps_data[index]["updated_at"] = datetime.utcnow().isoformat()

    try:
        print(f" Starting workflow for task {task_id}")
        
        for i, plan_step_title in enumerate(initial_plan_steps):
            print(f"\n{'='*60}")
            print(f"Step {i+1}/{len(initial_plan_steps)}: {plan_step_title[:80]}...")
            print(f"{'='*60}")
            
            update_step_status(i, "running", f"Executing: {plan_step_title}")

            # Execute step via appropriate agent
            try:
                actual_step_description, agent_name, output = executor_agent_step(
                    plan_step_title, 
                    execution_history, 
                    prompt
                )
                
                # Store in history
                execution_history.append([plan_step_title, actual_step_description, output])
                
                # Format substep for UI
                def esc(s: str) -> str:
                    return html.escape(s or "")

                substep_content = f"""
<div style='border:1px solid #e5e7eb; border-radius:8px; padding:16px; margin:8px 0; background:#ffffff;'>
  <div style='font-weight:600; color:#2563eb; margin-bottom:8px;'> User Prompt</div>
  <div style='background:#f3f4f6; padding:8px; border-radius:4px; margin-bottom:12px;'>
    {esc(prompt[:500])}{'...' if len(prompt) > 500 else ''}
  </div>

  <div style='font-weight:600; color:#16a34a; margin-bottom:8px;'> Context from Previous Steps</div>
  <div style='background:#f0fdf4; padding:8px; border-radius:4px; margin-bottom:12px; max-height:200px; overflow-y:auto;'>
    {esc(format_history(execution_history[-2:-1]) if len(execution_history) > 1 else 'First step - no prior context')}
  </div>

  <div style='font-weight:600; color:#f59e0b; margin-bottom:8px;'> Task Assigned</div>
  <div style='background:#fef3c7; padding:8px; border-radius:4px; margin-bottom:12px;'>
    {esc(actual_step_description)}
  </div>

  <div style='font-weight:600; color:#10b981; margin-bottom:8px;'> Agent Output</div>
  <div style='background:#f9fafb; padding:8px; border-radius:4px; max-height:400px; overflow-y:auto;'>
    {output}
  </div>
</div>
""".strip()
                
                update_step_status(
                    i,
                    "done",
                    f"Completed: {plan_step_title}",
                    {
                        "title": f"Executed by {agent_name}",
                        "content": substep_content,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                print(f" Step {i+1} completed successfully")
                
            except Exception as step_error:
                print(f" Error in step {i+1}: {step_error}")
                update_step_status(
                    i,
                    "error",
                    f"Error: {str(step_error)}",
                    {
                        "title": "Execution Error",
                        "content": f"<div style='color:#dc2626;'>{esc(str(step_error))}</div>",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                raise  # Re-raise to trigger outer exception handler

        # Extract final report (last step output)
        final_report_markdown = (
            execution_history[-1][-1] if execution_history else "No report generated."
        )

        result = {
            "html_report": final_report_markdown,
            "history": steps_data,
            "execution_summary": {
                "total_steps": len(initial_plan_steps),
                "completed_steps": len(execution_history),
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        # Update database with final results
        db = SessionLocal()
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task:
                task.status = "done"
                task.result = json.dumps(result)
                task.updated_at = datetime.utcnow()
                db.commit()
                print(f" Task {task_id} completed and saved to database")
        except Exception as db_error:
            print(f" Database update error: {db_error}")
            db.rollback()
        finally:
            db.close()

    except Exception as e:
        print(f" Workflow error for task {task_id}: {e}")
        
        # Mark failed step
        if steps_data:
            error_step_index = next(
                (i for i, s in enumerate(steps_data) if s["status"] == "running"),
                len(steps_data) - 1
            )
            if error_step_index >= 0:
                update_step_status(
                    error_step_index,
                    "error",
                    f"Workflow failed: {str(e)}",
                    {
                        "title": "Fatal Error",
                        "content": f"<div style='color:#dc2626; padding:12px; background:#fee2e2; border-radius:4px;'><strong>Error:</strong> {html.escape(str(e))}</div>",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )

        # Update database status
        db = SessionLocal()
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task:
                task.status = "error"
                task.updated_at = datetime.utcnow()
                task.result = json.dumps({
                    "error": str(e),
                    "history": steps_data
                })
                db.commit()
        except Exception as db_error:
            print(f" Error updating database: {db_error}")
            db.rollback()
        finally:
            db.close()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=True,
        log_level="info"
    )
