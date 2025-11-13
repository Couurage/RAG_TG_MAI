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
    def __init__(self, alias: str = "default"):
        self.alias = alias
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
            return Collection(MILVUS_COLLECTION, using=self.alias)

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
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
                dim=EMBED_DIM,
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

    def add_chunks(
        self,
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
            [doc_id] * n,
            chunk_ids,
            [source_path] * n,
            sections or ["" for _ in range(n)],
            contents,
            embeddings,
        ]

        insert_fields = [
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
    ):
        self.collection.load()
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16},
        }

        res = self.collection.search(
            data=[query_emb],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["doc_id", "chunk_id", "source_path", "section", "content"],
        )
        return res[0]
