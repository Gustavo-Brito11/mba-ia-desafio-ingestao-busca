# Desafio MBA Engenharia de Software com IA - Full Cycle

Resumo
- Projeto de busca vetorial + chat. PDFs são ingeridos em um banco Postgres com extensão pgvector; as consultas fazem busca por similaridade e o texto retornado é usado como contexto para o modelo de chat.

Pré-requisitos
- Docker & Docker Compose
- Python 3.10+ (recomendado)
- Credenciais/configuração do provedor de modelos/embeddings (ex.: Google Generative AI) se for o caso

Instalação das dependências
- Recomenda-se usar um virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Variáveis de ambiente (.env)
- Crie um arquivo `.env` na raiz do projeto com pelo menos as seguintes variáveis:

```
PDF_PATH=/caminho/para/seu.pdf
GOOGLE_EMBEDDING_MODEL=nome_do_modelo_de_embedding
PG_VECTOR_COLLECTION_NAME=colecao_vetorial
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag
# Se usar Google Generative AI, defina também a variável de credenciais, ex.:
# GOOGLE_APPLICATION_CREDENTIALS=/caminho/para/credentials.json
```

Ordem de execução
1. Subir o banco de dados (Postgres + pgvector):

```bash
docker compose up -d
```

Aguarde até o container do Postgres ficar saudável (`docker compose ps` ou `docker compose logs -f postgres`).

2. Executar a ingestão do PDF (vai carregar, dividir em chunks, gerar embeddings e inserir no banco):

```bash
python src/ingest.py
```

3. Rodar o chat (terminal interativo):

```bash
python src/chat.py
```

- No prompt, digite suas perguntas. Para sair, digite `sair`.

Observações e dicas
- Verifique se `DATABASE_URL` aponta para o mesmo host/porta do container Postgres.
- Se a busca não retornar contexto, confirme que `ingest.py` inseriu documentos na coleção definida por `PG_VECTOR_COLLECTION_NAME`.
- Ajuste as variáveis de modelo/credenciais conforme o provedor que estiver usando (Google, OpenAI, etc.).

Comandos úteis

```bash
# Ver logs do postgres
docker compose logs -f postgres

# Parar e remover containers
docker compose down
```

Licença
- Use conforme necessário.