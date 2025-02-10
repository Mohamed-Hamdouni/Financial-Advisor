# helper_utils.py
import os
import numpy as np
import chromadb
import pandas as pd
from PyPDF2 import PdfReader
from typing import Callable, Any, List  # Ajout de List ici
from pathlib import Path
import hashlib
import pickle


def transform_data(input_array, transformer):
    """Maps input data through dimensionality reduction"""
    return transformer.transform(input_array)


def format_string(content, line_length=80):
    """Formats text content into lines of specified length"""
    return "\n".join([content[i:i + line_length] for i in range(0, len(content), line_length)])


def parse_document(file_location):
    """Extracts content from PDF documents"""
    try:
        print(f"[INFO] Attempting to parse document: {file_location}")
        content = []
        with open(file_location, "rb") as source:
            doc = PdfReader(source)
            for page in doc.pages:
                text = page.extract_text()
                if text:  # Only append if text was successfully extracted
                    content.append(text)
        if not content:
            print(f"[WARNING] No content extracted from {file_location}")
            return ""
        return "\n".join(content)
    except Exception as e:
        print(f"[ERROR] Failed to parse {file_location}: {str(e)}")
        return ""


def initialize_database(source_file: str, db_name: str, embed_func: Callable[[str], Any], progress_callback: Callable[[str, float], None] = None):
    """Sets up vector database with document embeddings"""
    try:
        print(f"[INFO] Initializing database from source: {source_file}")
        source_path = Path(source_file)
        
        # Créer le dossier pour la base de données persistante avec les bonnes permissions
        db_path = Path("vector_db") / db_name
        db_path.mkdir(parents=True, exist_ok=True)
        # S'assurer que le dossier a les bonnes permissions
        os.chmod(str(db_path), 0o755)
        
        if progress_callback:
            progress_callback("Setting up database storage...", 0.2)
            
        # Initialiser le client Chroma avec persistance et settings spécifiques
        settings = chromadb.Settings(
            is_persistent=True,
            persist_directory=str(db_path),
            anonymized_telemetry=False
        )
        
        client = chromadb.PersistentClient(
            path=str(db_path),
            settings=settings
        )
        print(f"[INFO] Using persistent storage at: {db_path}")

        try:
            collection = client.get_collection(name=db_name)
            print(f"[INFO] Using existing collection: {db_name}")
            if progress_callback:
                progress_callback("Using existing database...", 1.0)
            return collection
        except Exception:
            if progress_callback:
                progress_callback("Creating new collection...", 0.5)
                
            print(f"[INFO] Creating new collection: {db_name}")
            collection = client.create_collection(name=db_name)
            
            if source_path.is_dir():
                print("[INFO] Processing PDF directory...")
                content = []
                pdf_files = list(source_path.glob("*.pdf"))
                total_files = len(pdf_files)
                
                for idx, pdf_file in enumerate(pdf_files):
                    try:
                        if progress_callback:
                            progress_val = 0.5 + (0.3 * (idx / total_files))
                            progress_callback(f"Processing PDF {idx + 1}/{total_files}: {pdf_file.name}", progress_val)
                            
                        print(f"[INFO] Processing: {pdf_file.name}")
                        doc_content = parse_document(str(pdf_file))
                        if doc_content:
                            content.append(doc_content)
                    except Exception as e:
                        print(f"[ERROR] Failed to process {pdf_file.name}: {e}")
                        continue
                
                if not content:
                    raise ValueError("No content could be extracted from any PDF files")
                    
                full_content = "\n\n".join(content)
                
                # Split content into smaller chunks
                print("[INFO] Splitting content into segments...")
                segments = [s for s in full_content.split("\n\n") if s.strip()]
                if not segments:
                    raise ValueError("No valid content segments found")
                print(f"[INFO] Created {len(segments)} segments")

                # Ajout d'un mécanisme de cache pour les embeddings
                cache_dir = Path("vector_db") / "cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                # Ajouter le cache des embeddings
                def cached_embed_query(text: str) -> List[float]:
                    """Cache embeddings for better performance"""
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    cache_file = cache_dir / f"{text_hash}.pkl"
                    
                    if cache_file.exists():
                        with open(cache_file, 'rb') as f:
                            return pickle.load(f)
                            
                    # Use encode instead of embed_query and take first result
                    embedding = embed_func.encode([text])[0].tolist()
                    
                    with open(cache_file, 'wb') as f:
                        pickle.dump(embedding, f)
                        
                    return embedding
                
                if progress_callback:
                    progress_callback("Creating embeddings...", 0.8)
                
                # Create embeddings and add to collection in batches
                batch_size = 100
                total_batches = (len(segments) - 1) // batch_size + 1
                for i in range(0, len(segments), batch_size):
                    if progress_callback:
                        batch_progress = i // batch_size
                        progress_val = 0.8 + (0.2 * (batch_progress / total_batches))
                        progress_callback(f"Adding batch {batch_progress + 1}/{total_batches}", progress_val)
                    
                    end_idx = min(i + batch_size, len(segments))
                    batch_segments = segments[i:end_idx]
                    batch_ids = [f"doc_{j}" for j in range(i, end_idx)]
                    
                    print(f"[INFO] Processing batch {i//batch_size + 1}/{(len(segments)-1)//batch_size + 1}")
                    
                    # Create embeddings for this batch
                    batch_embeddings = [cached_embed_query(seg) for seg in batch_segments]
                    
                    # Add to collection
                    collection.add(
                        documents=batch_segments,
                        embeddings=batch_embeddings,
                        ids=batch_ids
                    )
                    print(f"[INFO] Added batch {i//batch_size + 1}")

        print("[SUCCESS] Database initialization complete!")
        return collection

    except Exception as e:
        print(f"[ERROR] Database initialization failed: {str(e)}")
        # Afficher plus de détails sur l'erreur
        import traceback
        print(traceback.format_exc())
        raise
