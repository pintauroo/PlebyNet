# PlebyNet


Experiments in progress:

    |   Heterogeneous nodes |    Jobs duration |       measures             | Het JOBS
1           yes                   not fixed        JCT, first_rejected          yes
2           no                    not fixed        JCT, first_rejected
3           yes                    fixed           JCT, first_rejected
4           no                     fixed           JCT, first_rejected



docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' bmv2-container
ssh-keygen -R 172.17.0.2
ssh-keygen -R [localhost]:2222
ssh p4user@localhost -p 2222
ssh p4user@172.17.0.2 -p 2222
