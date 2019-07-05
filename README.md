# Trabalho da disciplina Progamação Paralela e Concorrente
Este repositório contêm a solução para o trabalho da disciplina de GCC-177.

## O grupo:
- Claudinei Moreira Silva: 
- Franciscone Luiz: https://github.com/franciscone14
- Igor Carvalho: https://github.com/igorecarvalho
- Victor Hugo Landin: https://github.com/vhal9

##O Trabalho 
- O problema esta descrito no arquivo trabalho-de-implementacao.txt.
- Arquivo main.py realiza o estudados dos dados.
- Arquivo Relatório_Trabalho_PPC.pdf contém o relatório de desenvolvimento do trabalho.

# Para executar:
- ```git clone https://github.com/franciscone14/trablho_ppc.git```
- entre no diretório do repositório.
- crie o diretório models
- instale as bibliotecas do python com o comando: ```pip install -r requirements.txt```
- execute o programa com o comando: ```mpiexec -n 8 python3 main.py```, onde:
- - ```8``` é o numero de processos.
- - ```main.py``` é o nome do programa.