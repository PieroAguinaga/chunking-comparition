-- 1. Extensión necesaria para embeddings
create extension if not exists vector;

-- 2. Tabla principal
create table if not exists documents (
  id            bigserial primary key,
  paper_id      text not null,
  method        text not null,          
  chunk_index   int not null,
  content       text not null,
  embedding     vector(1536) not null,
  metadata      jsonb default '{}'::jsonb,
  created_at    timestamp default now()
);

-- 3. Índice vectorial (búsqueda semántica)
create index if not exists documents_embedding_idx
  on documents
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- 4. Índices para filtros (muy importantes para benchmarking)
create index if not exists documents_paper_id_idx
  on documents (paper_id);

create index if not exists documents_method_idx
  on documents (method);

-- índice combinado (mejora queries reales)
create index if not exists documents_paper_method_idx
  on documents (paper_id, method);

-- 5. Función de búsqueda semántica
create or replace function match_documents(
  query_embedding  vector(1536),
  match_count      int,
  filter_paper_id  text default '',
  filter_method    text default ''
)
returns table (
  id          bigint,
  paper_id    text,
  method      text,
  chunk_index int,
  content     text,
  metadata    jsonb,
  similarity  float
)
language sql stable
as $$
  select
    id,
    paper_id,
    method,
    chunk_index,
    content,
    metadata,
    1 - (embedding <=> query_embedding) as similarity
  from documents
  where 
    (filter_paper_id = '' or paper_id = filter_paper_id)
    and (filter_method = '' or method = filter_method)
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- 6. (Opcional pero MUY recomendado) tabla de control de migraciones
create table if not exists schema_migrations (
  id text primary key,
  executed_at timestamp default now()
);