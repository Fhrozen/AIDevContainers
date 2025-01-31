ARG TAG
FROM ${TAG}:cpu-3.10

WORKDIR /opt

RUN git clone https://github.com/SamuelSchmidgall/AgentLaboratory
WORKDIR /opt/AgentLaboratory

COPY req.txt ./
RUN pip install -r req.txt && \
    rm -rf /root/.cache/pip && \
    rm ai_lab_repo.py

COPY ai_lab_repo.py /opt/AgentLaboratory

# USER fhrozen
# WORKDIR /home/fhrozen
