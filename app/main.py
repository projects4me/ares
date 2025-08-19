from fastapi import FastAPI 
import debugpy

# Create FastAPI instance
app = FastAPI(
    title="LLM API",
    description="A simple FastAPI LLM application",
    version="1.0.0"
)

# listen for debugpy
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger to attach...")