FROM registry.redhat.io/ubi8/python-39:latest

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt


COPY api.py ./api.py

#USER 1001
EXPOSE 8080


CMD ["python", "api.py", "8080"]
