from fastapi import FastAPI


app = FastAPI(title="Nexus Odonto ChatBot AI")


@app.get("/health")
def health_check() -> dict[str, str]:
	return {"status": "ok"}