FROM quay.io/modh/odh-generic-data-science-notebook@sha256:ebb5613e6b53dc4e8efcfe3878b4cd10ccb77c67d12c00d2b8c9d41aeffd7df5

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt


COPY api.py ./api.py

#USER 1001
EXPOSE 8080


CMD ["python", "api.py", "8080"]
