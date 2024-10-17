# PlebyNet


Experiments in progress:

    |   Heterogeneous nodes |    Jobs duration |       measures             | JOBS SIZE
1           yes                   not fixed        JCT, first_rejected          yes
2           no                    not fixed        JCT, first_rejected
3           yes                    fixed           JCT, first_rejected
4           no                     fixed           JCT, first_rejected


Grafico 1: 
Asse X: residual GPU/CPU capacity 
Asse Y: allocation ratio (Requested/Allocated)
Parametro: 10, 100, 500, 1000

Grafico 2: 
Asse X: avg size of the request 
Asse Y: Jain Index of residual capacity
Parametro: utility function


Grafico 3: Plottare la System Capacity over time



load balancing of gpu cpu resources at given time instants.

docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' bmv2-container
ssh-keygen -R 172.17.0.2
ssh-keygen -R [localhost]:2222
ssh p4user@localhost -p 2222
ssh p4user@172.17.0.2 -p 2222
