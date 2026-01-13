#!/bin/bash

echo "🧹 Iniciando limpieza profunda de archivos residuales..."

# 1. Borrar carpetas temporales y recrearlas vacías
echo "🗑️ Vaciando carpetas input/ y output/..."
rm -rf input/*
rm -rf output/*

# 2. Borrar archivos específicos que suelen quedar huérfanos
echo "🗑️ Eliminando archivos temporales .gguf y .safetensors en la raíz..."
rm -f *.safetensors
rm -f *.gguf
rm -f temp.gguf
rm -f output/temp.gguf

# 3. Limpiar la caché de pip (instalaciones de python)
echo "🗑️ Limpiando caché de pip..."
pip cache purge

# 4. LIMPIEZA CLAVE: Caché de Hugging Face
# HF guarda una copia de lo que subes en una carpeta oculta.
echo "🗑️ Limpiando caché de Hugging Face (descargas y subidas)..."
rm -rf ~/.cache/huggingface/hub/*

echo "✅ Limpieza completada."
df -h / | grep / # Mostrar espacio liberado