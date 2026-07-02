from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def root_route():
    return {'message': 'working!'}

@app.get('/health')
def health_route():
    return {'status':'ok'}
    