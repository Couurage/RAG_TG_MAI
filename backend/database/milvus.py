from __future__ import annotations

from typing import List, Optional

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from config import (
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_DB_NAME,
    MILVUS_COLLECTION,
    EMBED_DIM,
)


class Milvus:
    def __init__(self, alias: str = "default", embed_dim: Optional[int] = None):
        self.alias = alias
        self.embed_dim = embed_dim or EMBED_DIM
        connections.connect(
            alias,
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            user=None,
            password=None,
            db_name=MILVUS_DB_NAME,
        )
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        if utility.has_collection(MILVUS_COLLECTION, using=self.alias):
            collection = Collection(MILVUS_COLLECTION, using=self.alias)
            embed_field = next(
                (field for field in collection.schema.fields if field.name == "embedding"),
                None,
            )
            if embed_field and embed_field.params.get("dim") != self.embed_dim:
                raise ValueError(
                    f"Existing collection '{MILVUS_COLLECTION}' has dim={embed_field.params.get('dim')} "
                    f"but embedder reports dim={self.embed_dim}. Drop the collection or adjust config."
                )
            has_owner = any(field.name == "owner_id" for field in collection.schema.fields)
            if not has_owner:
                raise ValueError(
                    f"Existing collection '{MILVUS_COLLECTION}' lacks field 'owner_id'. "
                    "Drop the collection or migrate schema to continue."
                )
            return collection

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(
                name="owner_id",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="doc_id",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="chunk_id",
                dtype=DataType.INT32,
            ),
            FieldSchema(
                name="source_path",
                dtype=DataType.VARCHAR,
                max_length=512,
            ),
            FieldSchema(
                name="section",
                dtype=DataType.VARCHAR,
                max_length=256,
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=4096,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embed_dim,
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="RAG chunks for TG bot",
        )
        coll = Collection(
            name=MILVUS_COLLECTION,
            schema=schema,
            using=self.alias,
        )

        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 1024},
        }
        coll.create_index(
            field_name="embedding",
            index_params=index_params,
        )

        coll.load()
        return coll

    def clear(self):
        """
        Полностью очищает коллекцию, но сохраняет её схему и индексы.
        """
        print(f"⚠️ Удаляю все данные из коллекции '{MILVUS_COLLECTION}'...")

        # Удаляем все строки по фильтру (условие всегда True)
        self.collection.delete(expr="id >= 0")
        self.collection.flush()

        print("✅ Коллекция очищена")

    def add_chunks(
        self,
        owner_id: Optional[int],
        doc_id: int,
        source_path: str,
        sections: List[Optional[str]],
        contents: List[str],
        embeddings: List[List[float]],
    ) -> List[int]:
        """
        doc_id       — одинаковый для всех чанков одного документа
        source_path  — путь к исходному файлу
        sections     — список заголовков (можно None)
        contents     — тексты чанков
        embeddings   — эмбеддинги чанков (len == len(contents))
        """

        assert len(contents) == len(embeddings)
        if sections and len(sections) != len(contents):
            raise ValueError("sections и contents разной длины")

        n = len(contents)
        chunk_ids = list(range(n))

        data = [
            [owner_id] * n,
            [doc_id] * n,
            chunk_ids,
            [source_path] * n,
            sections or ["" for _ in range(n)],
            contents,
            embeddings,
        ]

        insert_fields = [
            "owner_id",
            "doc_id",
            "chunk_id",
            "source_path",
            "section",
            "content",
            "embedding",
        ]

        res = self.collection.insert(data, insert_fields=insert_fields)
        self.collection.flush()
        return res.primary_keys

    def search(
        self,
        query_emb: List[float],
        top_k: int = 5,
        *,
        doc_id: Optional[int] = None,
        source_path: Optional[str] = None,
        owner_id: Optional[int] = None,
    ):
        self.collection.load()
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16},
        }

        expr_parts = []
        if doc_id is not None:
            expr_parts.append(f"doc_id == {int(doc_id)}")
        elif source_path is not None:
            safe_path = str(source_path).replace("\\", "\\\\").replace('"', '\\"')
            expr_parts.append(f'source_path == "{safe_path}"')
        if owner_id is not None:
            expr_parts.append(f"owner_id == {int(owner_id)}")
        expr = " and ".join(expr_parts) if expr_parts else None

        res = self.collection.search(
            data=[query_emb],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["doc_id", "chunk_id", "source_path", "section", "content"],
        )
        return res[0]

    def delete_doc(
        self,
        doc_id: Optional[int] = None,
        *,
        source_path: Optional[str] = None,
        owner_id: Optional[int] = None,
    ) -> int:
        """
        Удаляет все чанки документа по doc_id или source_path.
        Возвращает число удалённых записей.
        """
        self.collection.load()
        if doc_id is None and source_path is None:
            raise ValueError("doc_id or source_path must be specified")

        expr_parts = []
        if doc_id is not None:
            expr_parts.append(f"doc_id == {int(doc_id)}")
        elif source_path is not None:
            safe_path = str(source_path).replace("\\", "\\\\").replace('"', '\\"')
            expr_parts.append(f'source_path == "{safe_path}"')
        if owner_id is not None:
            expr_parts.append(f"owner_id == {int(owner_id)}")

        expr = " and ".join(expr_parts)

        result = self.collection.delete(expr=expr)
        deleted = getattr(result, "delete_count", 0)
        if deleted:
            self.collection.flush()
        return deleted
