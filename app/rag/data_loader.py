import argparse
import hashlib
from pathlib import Path

from langchain_community.document_loaders import JSONLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.agents.tools.qdrant_tool import get_vector_store
from app.core.config import settings


def load_source(source_path: Path):
	"""Lee un archivo de texto o JSON y devuelve documentos de LangChain."""
	if source_path.suffix.lower() == ".json":
		# Cada elemento del arreglo JSON se convierte en un documento.
		loader = JSONLoader(
			file_path=str(source_path),
			jq_schema=".[]",
			text_content=False,
		)
	elif source_path.suffix.lower() in {".txt", ".md"}:
		loader = TextLoader(str(source_path), encoding="utf-8")
	else:
		raise ValueError("El archivo debe tener extension .txt, .md o .json")

	return loader.load()


def split_documents(documents):
	"""Divide los documentos en fragmentos adecuados para la búsqueda semántica."""
	splitter = RecursiveCharacterTextSplitter(
		chunk_size=800,
		chunk_overlap=120,
		separators=["\n\n", "\n", ". ", " ", ""],
	)
	return splitter.split_documents(documents)


def build_document_ids(documents: list) -> list[str]:
	"""Genera IDs estables para que una segunda carga no duplique vectores."""
	return [
		hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
		for document in documents
	]


def load_clinical_knowledge(source_path: Path) -> int:
	"""Vectoriza e inserta el catálogo en la colección clínica de Qdrant."""
	documents = split_documents(load_source(source_path))
	if not documents:
		raise ValueError(f"No se encontraron documentos en {source_path}")

	# get_vector_store reutiliza la conexión y crea la colección si hace falta.
	vector_store = get_vector_store()
	vector_store.add_documents(documents, ids=build_document_ids(documents))
	return len(documents)


def main() -> None:
	"""Procesa los argumentos de consola y ejecuta la carga una sola vez."""
	parser = argparse.ArgumentParser(
		description="Carga conocimiento clínico en Qdrant."
	)
	parser.add_argument(
		"--source",
		type=Path,
		required=True,
		help="Ruta a un archivo .txt, .md o .json con el catálogo clínico.",
	)
	args = parser.parse_args()

	if not args.source.is_file():
		raise FileNotFoundError(f"No existe el archivo: {args.source}")

	loaded_count = load_clinical_knowledge(args.source)
	print(
		f"Se cargaron {loaded_count} fragmentos en "
		f"'{settings.qdrant_collection_name}'."
	)


if __name__ == "__main__":
	main()


"""Para cargar el catálogo clínico en Qdrant.

Se debe Usar:
	python -m app.rag.data_loader --source data/catalogo_clinico.txt
	python -m app.rag.data_loader --source data/catalogo_clinico.json
"""
