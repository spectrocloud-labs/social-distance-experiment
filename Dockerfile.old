FROM python:3.9

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir schedule

COPY . .

EXPOSE 8080/tcp

CMD [ "python", "Run.py" ]
