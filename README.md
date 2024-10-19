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


VD: Variable Duration
HN: Heterogeneous Nodes
ND: No Discard
BW: Bandwidth



Devo rifare stesso test scalando numero di nodi e di job senza modificare piu il codice
raccolgo tutti i dati senza la banda e poi facciamo vedere come peggiorano le cose con la banda
non rimane da capire come implemenlo possiamo fare comae tiresiastare lo speedup della banda dato che 
ora dobbiamo rifare stessi test con la banda!
ora li rifacciamo scalati con 100 nodi e 200 jobs in settings non fix duration, het nodes  no discard jobs without bw


Test 1 50 nodes 100 jobs non fix duration, het nodes  no discard jobs 
50N_100J_NFD_HN_NDJ
Test 1 100 nodes 200 jobs non fix duration, het nodes  no discard jobs 
100N_200J_NFD_HN_NDJ

