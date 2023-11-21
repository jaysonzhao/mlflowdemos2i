FROM jupyter/scipy-notebook

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt


COPY api.py ./api.py

#USER 1001
EXPOSE 8080

RUN python3 train.py
CMD ["python3", "api.py", "8080"]