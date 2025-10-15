"""
=================================================================================
CHATBOT JURIDIQUE MAROCAIN - ELYSIA AI
=================================================================================
Version corrigée avec noms de propriétés en camelCase
=================================================================================
"""

import pickle
import pandas as pd
import json
import datetime
import socket
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure

socket.setdefaulttimeout(300)

# =============================================================================
# ÉTAPE 1 : CHARGEMENT DU DATASET
# =============================================================================
print("=" * 80)
print("ÉTAPE 1 : CHARGEMENT DU DATASET")
print("=" * 80)

with open("sample_3k_legal_texts_15sept2025.pickle", "rb") as f:
    data = pickle.load(f)
print(f"✓ {len(data)} documents chargés\n")

# =============================================================================
# ÉTAPE 2 : CONVERSION EN DATAFRAME
# =============================================================================
print("=" * 80)
print("ÉTAPE 2 : CONVERSION EN DATAFRAME")
print("=" * 80)

df = pd.DataFrame(data)
print(f"✓ {df.shape[0]} lignes × {df.shape[1]} colonnes\n")

# =============================================================================
# ÉTAPE 3 : NETTOYAGE DES DONNÉES
# =============================================================================
print("=" * 80)
print("ÉTAPE 3 : NETTOYAGE DES DONNÉES")
print("=" * 80)

# Remplacement des NaN
df = df.fillna("")

# Normalisation des dates
date_column = 'Date' if 'Date' in df.columns else 'date' if 'date' in df.columns else None

if date_column:
    df['date'] = pd.to_datetime(df[date_column], errors='coerce')
    df['date_string'] = df['date'].dt.strftime('%Y-%m-%d').fillna("")
    df['year'] = df['date'].dt.year.fillna(0).astype(int)
    df['month'] = df['date'].dt.month.fillna(0).astype(int)
else:
    df['date_string'] = ""
    df['year'] = 0
    df['month'] = 0

# Standardisation des colonnes - IMPORTANT: en camelCase pour Elysia
column_mapping = {
    'lawNumber': ['LawNumber', 'law_number', 'numero_loi'],
    'documentType': ['DocumentType', 'document_type', 'type_document'],
    'subject': ['Subject', 'subject', 'sujet'],
    'signatures': ['Signatures', 'signatures', 'signataires'],
    'keywords': ['Keywords', 'keywords', 'mots_cles'],
    'pageNumber': ['PageNumber', 'page_number', 'numero_page'],
    'summary': ['Summary', 'summary', 'resume'],
    'text': ['text', 'law_text', 'LawText', 'texte'],
    'references': ['references', 'references_to_other_laws', 'ReferencesToOtherLaws']
}

for standard_name, possible_names in column_mapping.items():
    found = False
    for col_name in possible_names:
        if col_name in df.columns:
            df[standard_name] = df[col_name].astype(str).str.strip().replace('nan', '')
            found = True
            break
    if not found:
        df[standard_name] = ""

print("✓ Nettoyage terminé\n")

# =============================================================================
# ÉTAPE 4 : CRÉATION DU TEXTE COMPLET POUR EMBEDDING
# =============================================================================
print("=" * 80)
print("ÉTAPE 4 : CRÉATION DU TEXTE COMPLET POUR EMBEDDING")
print("=" * 80)

def create_full_text(row):
    parts = []
    if row.get('lawNumber', ''): parts.append(f"Numéro: {row['lawNumber']}")
    if row.get('documentType', ''): parts.append(f"Type: {row['documentType']}")
    if row.get('date_string', ''): parts.append(f"Date: {row['date_string']}")
    if row.get('subject', ''): parts.append(f"Sujet: {row['subject']}")
    if row.get('signatures', ''): parts.append(f"Signataires: {row['signatures']}")
    if row.get('summary', ''): parts.append(f"Résumé: {row['summary']}")
    if row.get('keywords', ''): parts.append(f"Mots-clés: {row['keywords']}")
    if row.get('text', ''): parts.append(f"Texte: {row['text']}")
    return "\n".join(parts)

df['full_text'] = df.apply(create_full_text, axis=1)
print("✓ Champ 'full_text' créé\n")

# =============================================================================
# ÉTAPE 5 : SAUVEGARDE DES FICHIERS
# =============================================================================
print("=" * 80)
print("ÉTAPE 5 : SAUVEGARDE DES FICHIERS")
print("=" * 80)

# CSV
df.to_csv("dataset_nettoye.csv", index=False, encoding='utf-8-sig')

# JSON
documents = df.to_dict('records')
for doc in documents:
    for key, value in list(doc.items()):
        if isinstance(value, (list, dict)):
            continue
        elif isinstance(value, (pd.Timestamp, datetime.datetime)):
            doc[key] = value.isoformat()
        elif value is None or (isinstance(value, float) and pd.isna(value)):
            doc[key] = ""
        elif isinstance(value, str) and value in ['nan', 'NaN', 'None']:
            doc[key] = ""

with open("documents_pour_weaviate.json", 'w', encoding='utf-8') as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

# Statistiques
stats = {
    "total_documents": len(df),
    "types_documents": {k: int(v) for k, v in df['documentType'].value_counts().items() if k},
    "annees": {str(int(k)): int(v) for k, v in df['year'].value_counts().sort_index().items() if k > 0}
}

with open("dataset_statistiques.json", 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print("✓ Fichiers sauvegardés\n")

# =============================================================================
# ÉTAPE 6 : CONNEXION À WEAVIATE
# =============================================================================
print("=" * 80)
print("ÉTAPE 6 : CONNEXION À WEAVIATE")
print("=" * 80)

client = weaviate.connect_to_weaviate_cloud(
    cluster_url="https://yf9yqnuqqxid8j1qflpcvg.c0.us-west3.gcp.weaviate.cloud",
    auth_credentials=Auth.api_key("bVZnLyt5d280VEpzMlJxQV94L0FTMjFlS1g2OUhhc1ZENm5pZE1ab3haSEVFNS84SFRUMXpWQWNvckM4PV92MjAw"),
)
print(f"✓ Connexion établie\n")

# =============================================================================
# ÉTAPE 7 : CRÉATION DU SCHÉMA (CORRIGÉ EN camelCase)
# =============================================================================
print("=" * 80)
print("ÉTAPE 7 : CRÉATION DU SCHÉMA")
print("=" * 80)

if "LegalDocument" in client.collections.list_all():
    user_input = input("Collection existe. Supprimer et recréer ? (o/n) : ").strip().lower()
    if user_input == 'o':
        client.collections.delete("LegalDocument")
        print("✓ Collection supprimée")

if "LegalDocument" not in client.collections.list_all():
    client.collections.create(
        name="LegalDocument",
        description="Documents juridiques marocains",
vectorizer_config=Configure.Vectorizer.text2vec_openai(
    model="text-embedding-3-large",
),
        properties=[
            # IMPORTANT: Propriétés en camelCase pour correspondre à Elysia
            Property(name="lawNumber", data_type=DataType.TEXT),
            Property(name="documentType", data_type=DataType.TEXT),
            Property(name="date_string", data_type=DataType.TEXT),
            Property(name="year", data_type=DataType.INT),
            Property(name="subject", data_type=DataType.TEXT),
            Property(name="signatures", data_type=DataType.TEXT),
            Property(name="summary", data_type=DataType.TEXT),
            Property(name="keywords", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="full_text", data_type=DataType.TEXT),
            Property(name="pageNumber", data_type=DataType.TEXT),
            Property(name="references", data_type=DataType.TEXT),
        ]
    )
    print("✓ Collection créée avec propriétés en camelCase")

collection = client.collections.get("LegalDocument")
print("✓ Schéma créé\n")

# =============================================================================
# ÉTAPE 8 : CHARGEMENT DU MODÈLE D'EMBEDDING
# =============================================================================
print("=" * 80)
print("ÉTAPE 8 : CHARGEMENT DU MODÈLE D'EMBEDDING")
print("=" * 80)

model = SentenceTransformer("intfloat/multilingual-e5-large")
print("✓ Modèle chargé\n")

# =============================================================================
# ÉTAPE 9 : INGESTION DES DOCUMENTS
# =============================================================================
print("=" * 80)
print("ÉTAPE 9 : INGESTION DES DOCUMENTS")
print("=" * 80)

batch_size = 50
total_ingested = 0
failed = 0

for i in tqdm(range(0, len(documents), batch_size), desc="Progression"):
    batch = documents[i:i + batch_size]
    
    with collection.batch.dynamic() as batcher:
        for doc in batch:
            try:
                text_to_embed = doc.get("full_text", "")
                if not text_to_embed.strip():
                    failed += 1
                    continue
                
                vector = model.encode(text_to_embed, normalize_embeddings=True).tolist()
                
                # Gestion robuste du year
                try:
                    year_value = doc.get("year", 0)
                    year_value = 0 if year_value in ["", "nan", None] or pd.isna(year_value) else int(float(year_value))
                except (ValueError, TypeError):
                    year_value = 0
                
                # IMPORTANT: Propriétés en camelCase
                properties = {
                    "lawNumber": str(doc.get("lawNumber", ""))[:500],
                    "documentType": str(doc.get("documentType", ""))[:200],
                    "date_string": str(doc.get("date_string", ""))[:50],
                    "year": year_value,
                    "subject": str(doc.get("subject", ""))[:500],
                    "signatures": str(doc.get("signatures", ""))[:500],
                    "summary": str(doc.get("summary", ""))[:2000],
                    "keywords": str(doc.get("keywords", ""))[:500],
                    "text": str(doc.get("text", "")),
                    "full_text": str(doc.get("full_text", "")),
                    "pageNumber": str(doc.get("pageNumber", ""))[:50],
                    "references": str(doc.get("references", ""))[:1000],
                }
                
                batcher.add_object(properties=properties, vector=vector)
                total_ingested += 1
                
            except Exception as e:
                failed += 1
                continue

print(f"\n✓ Ingestion terminée : {total_ingested} succès, {failed} échecs\n")

# =============================================================================
# ÉTAPE 10 : VÉRIFICATION FINALE
# =============================================================================
print("=" * 80)
print("ÉTAPE 10 : VÉRIFICATION FINALE")
print("=" * 80)

total_in_db = collection.aggregate.over_all(total_count=True).total_count
print(f"✓ Documents dans Weaviate : {total_in_db}")

# Vérification du schéma
config = collection.config.get()
print(f"\n✓ Vérification du schéma:")
print(f"  Propriétés créées (en camelCase):")
for prop in config.properties:
    print(f"    - {prop.name}")

# Test de recherche
if total_in_db > 0:
    test_vector = model.encode("قانون البيئة", normalize_embeddings=True).tolist()
    results = collection.query.near_vector(near_vector=test_vector, limit=3)
    print(f"\n✓ Test de recherche : {len(results.objects)} résultats trouvés")

print("\n" + "=" * 80)
print(" SCRIPT TERMINÉ AVEC SUCCÈS")

client.close()