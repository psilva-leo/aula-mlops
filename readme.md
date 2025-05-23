# Aula MLOPs

## Notebooks

A pasta `notebooks` contém dois notebooks, sendo o `eda` responsável pela análise e preparação dos dados e `train` responsável pelo treino do modelo.

Antes de começar, instale as dependências do projeto
```bash
pip install --no-cache-dir -r requirements-dev.txt
```

Depois execute o notebook `eda`, pode roda-lo por completo, mas recomendo que explore, teste e crie novas transformações e análises.
O notebook `train` trabalha o treino e seleção dos modelos. É nele que é feito testes com diferentes algoritmos e hyper-parâmetros.

## Pipeline

Notebooks são ótimos, mas não são artefatos ou códigos produtivos. É necessários transformá-los em scripts produtivos, 
rastreáveias e que podem ser rodados facilmente por outras pessoas ou processos em outros lugares.

Em `training_pipelines` vemos essa transformação dos notebooks em pipelines produtivos. Veja como o mesmo código tem ligeiras alterações na estrutura e forma.

Antes de executar o pipeline, é necessário subir o mlflow para rastrear o experimento e versionar o modelo treinado.
```bash
docker compose up
```

Para executá o pipeline, basta rodar `python pipeline.py`.

Você verá um experimento chamado `house_pricing` com uma execução (run). Nele tem informações como duração da execução,
o criador, tags, código primário, modelo verionado, parâmetros utilizados, métricas  e artefatos.

## Serving

Agora podemos usar o modelo em produção!
A API subirá automaticamnte com o docker compose, mas se quiser pode fazer manualmente também.

### Docker Compose
```bash
docker compose up
```

### Manualmente

```bash
docker build -t aula-mlops:0.1.0
```

## Teste

A API ficará disponível localmente (localhost) na porta 8080 expondo o enpoint /infer.

```curl
curl --location 'http://localhost:8080/infer' \
--header 'Content-Type: application/json' \
--data '{
    "Overall Qual": 1,
    "Exter Qual": "TA",
    "Bsmt Qual": "TA",
    "Total Bsmt SF": "1000",
    "1st Flr SF": "1656",
    "Gr Liv Area": "1656",
    "Kitchen Qual": "TA",
    "Garage Cars": "2",
    "Garage Area": "528"
}'
```
