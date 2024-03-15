import streamlit as st
import PyPDF2
from PIL import Image
from pdf2image import convert_from_path
from pdf2image import convert_from_bytes
import easyocr
import openai
import tempfile
import os
import time
import os
from dotenv import load_dotenv
import cv2
import numpy as np
from unidecode import unidecode
import threading
import concurrent.futures
import time
import pytesseract


# Función para procesar las imágenes extraídas
def process_images(images,ruta_carpeta, n):

    processed_images = []

    # Convertir la imagen a escala de grises
    #imagen = cv2.imread(images)
    #gray = cv2.cvtColor(cv2.imread(images), cv2.COLOR_BGR2GRAY)

    # Aplicar umbralización
    _, thresh = cv2.threshold(cv2.cvtColor(cv2.imread(images), cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Aplicar filtrado de suavizado
    #blur = cv2.medianBlur(thresh, 3)

    # Detectar bordes
    #edges = cv2.Canny(cv2.medianBlur(thresh, 3), 50, 150)

    # Dilatar los bordes para cerrar cualquier hueco
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    #dilate = cv2.dilate(cv2.Canny(cv2.medianBlur(thresh, 3), 50, 150), cv2.getStructuringElement(cv2.MORPH_RECT, (10,10)), iterations=2)


    # Aplicar detección de componentes conectados
    connected_components_image,n = connected_component_detection(cv2.dilate(cv2.Canny(cv2.medianBlur(thresh, 3), 50, 150), cv2.getStructuringElement(cv2.MORPH_RECT, (10,10)), iterations=2),cv2.imread(images),n, ruta_carpeta)

    processed_images.append(connected_components_image)

    return processed_images,n

# Función para determinar si un componente conectado cumple con los criterios de selección
def check_component_criteria(area, width, height, min_aspect_ratio, max_aspect_ratio, min_area_threshold):
    # Calcular la relación de aspecto del componente conectado
    aspect_ratio = width / float(height)

    # Verificar si el componente conectado cumple con los criterios de selección
    if (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio) and (area >= min_area_threshold):
        return True
    else:
        return False

# Función para detectar componentes conectados en una imagen binaria
def connected_component_detection(binary_image, imagen, n, ruta_carpeta, min_aspect_ratio=0, max_aspect_ratio=100, min_area_threshold=2500):
    # Encontrar componentes conectados en la imagen binaria
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    # Crear una copia de la imagen original para resaltar los componentes conectados
    highlighted_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    # Resaltar los componentes conectados en la imagen original

    for i in range(1, num_labels):

        # Extraer el área, el ancho, el alto y las coordenadas del componente conectado
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]

        # Verificar si el componente conectado cumple con los criterios de selección
        if check_component_criteria(area, width, height, min_aspect_ratio, max_aspect_ratio, min_area_threshold):
            # Dibujar un rectángulo alrededor del componente conectado en la imagen original
            cv2.rectangle(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR), (x, y), (x + width, y + height), (0, 255, 0), 2)

            imag=imagen.copy()
            roi = imag[y:y+height, x:x+width]
            cv2.imwrite(ruta_carpeta + "/" + 'recorte_{}.jpg'.format(n), roi)
            n = n + 1
    return highlighted_image,n



# Función para extraer texto de una imagen utilizando EasyOCR
def extraer_texto_de_imagen(imagen): # Dada una imagen, deveulve el texto que hay en ella
    #Por cada imagen que se le pase, extrae el texto usando easyOCR
    # Crea un objeto EasyOCR y especifica el idioma al español
    #lector = easyocr.Reader(["es"],gpu=True)
    texto = ""
    texto = pytesseract.image_to_string(imagen)
    return texto

def sacar_info_fact_openAI_cabecera(factura_cabecera):
  contexto = "A continuación te detallo la cabecera de una factura o albarán: "
  pregunta = "Extrae en formato tabla los datos del proveedor y añade como campo adicional si se trata de una factura o un albarán. En caso de no encontrar esta información no muestres nada."
  #Se repite hasta que haga todas las preguntas
  #Va de 3 en 3 porque la cuenta gratuita solo permite 3 peticiones por minuto
  #selecciona las 3 preguntas
  prompt = contexto + factura_cabecera + "\n" + pregunta

  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-0613",
      messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": prompt}
      ],
      temperature=0.5,
      max_tokens=300
  )
  #Muestra la informacion solicitada por cada pregunta
  st.write(response['choices'][0]['message']['content'])



def sacar_info_fact_openAI_resto(factura_resto):
  contexto_resto = "A continuación se muestran los campos de una factura o albarán junto con la informacion de estos campos donde además, trás esta información aparecen detalles extras de la factura o albarán."

  preguntas = ["Muestra en una tabla los siguientes campos para cada artículo. Código de Articulo (Suele ser un codigo alfanumerico sin espacios), Nombre o descripción del articulo, Unidades o subunidades del artiuclo, Precio o importe, Descuento (número que suele venir acompañado de un signo negativo), Impuestos o tasas. Alguno de estos campos podría ser vacio.",
             "En caso de tratarse de una factura, muestra en una tabla cual ha sido el valor total o cuantia y el impuesto total aplicado a la misma. Alguno de estos campos puede no especificarse."]
  texto = ""
  for pregunta in preguntas:

    prompt = contexto_resto + "\n" + factura_resto + pregunta

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=300
    )
    #Muestra la informacion solicitada por cada pregunta
    st.write(response['choices'][0]['message']['content'])

#Dado un string (en este caso el texto que tiene la factura) y una lista de preguntas definida anteriormente
#Utiliza la api de Openai para realizar las preguntas sobre ese texto
def sacar_info_fact_openAI(factura_txt, preguntas):
    contexto = "A continuación te detallo una factura o albarán que ha sido procesada de pdf a texto plano utilizando librerías de OCR, en alguna parte de la factura o albaran se encuentra una tabla principal cuyos campos están definidos previamente a los productos o artículos:"
    for pregunta in preguntas:
      prompt = contexto + factura_txt + "\nPregunta: " + pregunta
      response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo-0613",
          messages=[
              {"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": prompt}
          ],
          temperature=0.5,
          max_tokens=300
      )
      #Muestra la informacion solicitada por cada pregunta
      st.write(response['choices'][0]['message']['content'])



def recortar_horizontalmente(imagen, y_ini, recortes_h):
  #_, ancho = imagen.shape[:2]
  # Recortar desde el borde superior hasta la coordenada y
  #imagen_original = cv2.imread(imagen)
  imagen_cabecera = cv2.imread(imagen)[1:y_ini-3, :]
  imagen_resto = cv2.imread(imagen)[y_ini-3:, :]

  cv2.imwrite(recortes_h + "/" + 'recorte_cabecera.jpg', imagen_cabecera)
  cv2.imwrite(recortes_h + "/" + 'recorte_resto.jpg', imagen_resto)

def obtener_campos(ruta_imagen):
  imagen=cv2.imread(ruta_imagen)
  resultados = pytesseract.image_to_data(imagen, output_type=pytesseract.Output.DICT)
  # Iterar sobre los resultados para encontrar la palabra deseada
  x_rel = 0
  y_rel = 5
  h = 0
  for i, word in enumerate(resultados['text']):
      res_sinTildes= unidecode(word).lower()
      if ('descripcio' in res_sinTildes or
      'art.' == res_sinTildes or
      'art' == res_sinTildes or
      'articulo' in res_sinTildes or
      'quantitat' in res_sinTildes or
      'cantidad' in res_sinTildes or
      "cant" == res_sinTildes or
      'unidades' in res_sinTildes or
      'unids' == res_sinTildes or
      'detalle' in res_sinTildes or
      "desc." == res_sinTildes or
      "desc" == res_sinTildes or
      "imp." == res_sinTildes or
      "imp" == res_sinTildes or
      "precio" in res_sinTildes or
      "preu" == res_sinTildes):
          x_rel = resultados['left'][i]
          y_rel = resultados['top'][i]
          w = resultados['width'][i]
          h = resultados['height'][i]
          break
  return y_rel



def factura_a_txt(pdf_path): # Dado un pdf, devuelve todo el texto que hay en el
  #Lista de preguntas que se van a realizar a openai
  preguntas = ["Extrae en formato tabla los datos del proveedor y añade como campo adicional si se trata de una factura o un albarán. En caso de no encontrar esta información no muestres nada.",
              "Muestra en una tabla los siguientes campos para cada artículo. Código de Articulo (Suele ser un codigo alfanumerico sin espacios), Nombre o descripción del articulo, Unidades o subunidades del artiuclo, Precio o importe, Descuento (número que suele venir acompañado de un signo negativo), Impuestos o tasas. Alguno de estos campos podría ser vacio.",
              "En caso de tratarse de una factura, muestra en una tabla cual ha sido el valor total o cuantia y el impuesto total aplicado a la misma. Alguno de estos campos puede no especificarse."]
  texto = ""
  #Primero se comprueba si se puede extraer el texto con PyPDF2, es decir, si el PDF tiene texto seleccionable
  reader = PyPDF2.PdfReader(pdf_path)
  #Se va extrayendo el texto de cada una de las paginas del PDF
  for pagina in range(len(reader.pages)):
    texto += reader.pages[pagina].extract_text()
  #Si texto =="" quiere decir que el PDF no tenia texto seleccionable, es decir, era solo imagenes entonces hay que usar EasyOCR
  if texto == "": # No se ha podido obtener la info (PDF sin texto)
    #Se crea un directorio temporal para poder guardar las imagenes que se van a sacar del PDF
    with tempfile.TemporaryDirectory() as tmp_dir:
      #Convierte cada pagina del PDF en una imagen
      imagenes = convert_from_path(pdf_path, output_folder=tmp_dir)
    texto_cabecera=""
    texto_resto=""
    hilos_imagenes = []
    resultados = []
    with concurrent.futures.ThreadPoolExecutor(max_workers = 1) as executor_imagenes:
      for i, imagen in enumerate(imagenes):
        hilo_imagen = executor_imagenes.submit(procesar_pagina, imagen, i)
        hilos_imagenes.append(hilo_imagen)
      for h in concurrent.futures.as_completed(hilos_imagenes):
        resultados.append (h.result())
    array_ordenado = sorted(resultados, key=lambda x: x[2])
    for num_pagina in array_ordenado:
      texto_cabecera += num_pagina[0]
      texto_resto += num_pagina[1]
    sacar_info_fact_openAI_cabecera(texto_cabecera)
    sacar_info_fact_openAI_resto(texto_resto)
  else: # Se ha podido obtener el texto, se realiza la pregunta a openAI
    sacar_info_fact_openAI(texto, preguntas)

def procesa_cabecera(ruta_recortes, n):
  texto = ""
  for indice in range (n-1):
    nombre_archivo=os.listdir(ruta_recortes)[indice]
    ruta_archivo = os.path.join(ruta_recortes, nombre_archivo)
    if os.path.isfile(ruta_archivo):
      texto += extraer_texto_de_imagen(ruta_archivo)
  return texto

def procesar_pagina(imagen, i):
  texto_cabecera=""
  texto_campos=""
  texto_resto=""
  #Guardamos cada imagen
  nombre_imagen = f"pagina_{i+1}.jpg"
  imagen.save(nombre_imagen, "JPEG")
  #Guarda el texto de cada imagen
  with tempfile.TemporaryDirectory() as recortes_h:
    #min_y=obtener_coord_campos(nombre_imagen,reader)
    min_y=obtener_campos(nombre_imagen)
    recortar_horizontalmente(nombre_imagen,min_y, recortes_h)
    with tempfile.TemporaryDirectory() as ruta_recortes:
      n = 1
      ruta_cabecera = os.path.join(recortes_h, "recorte_cabecera.jpg")
      if os.path.isfile(ruta_cabecera):
        processed_images,n = process_images(ruta_cabecera,ruta_recortes,n)
      sorted(os.listdir(ruta_recortes))

      with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        hilo_cabecera = executor.submit(procesa_cabecera, ruta_recortes, n)
            #texto_cabecera = texto_cabecera + extraer_texto_de_imagen(ruta_archivo) + "\n"

        with concurrent.futures.ThreadPoolExecutor(max_workers = 1) as executor_resto:

          ruta_resto = os.path.join(recortes_h, "recorte_resto.jpg")
          if os.path.isfile(ruta_resto):
              hilo_resto = executor.submit(extraer_texto_de_imagen, ruta_resto)
              #texto_resto = texto_resto + extraer_texto_de_imagen(ruta_resto) + "\n"
          texto_cabecera += hilo_cabecera.result()
          texto_resto += hilo_resto.result()
  return texto_cabecera, texto_resto, i


def main():
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    # Carga la contraseña de openai del fichero .env
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    st.title("Procesamiento de Facturas")
    st.image("https://miro.medium.com/v2/resize:fit:4800/format:webp/1*fRYU2gXNPg8Q5wq8LLmQVw.png")

    # Pone el widget para poder subir un archivo, unicamente en formato pdf
    archivo_pdf = st.file_uploader("Selecciona un archivo PDF", type="pdf", accept_multiple_files= False)

    #Si el archivo no es nulo (si ha introducido ya un archivo) se procesa ese archivo
    if archivo_pdf is not None:
        # Guarda el archivo PDF temporalmente
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(archivo_pdf.read())
            nombre_pdf_temporal = tmp_file.name
        with tempfile.TemporaryDirectory() as tmp_dir:
            #Convierte cada pagina del PDF en una imagen
            imagenes = convert_from_path(nombre_pdf_temporal, output_folder=tmp_dir)
            
        #Muestra cada pagina del PDF como una imagen, con un pie de pagina que indica que pagina es
        #Se muestra en el lateral, a modo de sidebar
        st.markdown("""
          <style>
              .sidebar.content {
                  margin-left: 50%; /* Empieza en la mitad de la página */
                  transform: translateX(-50%); /* Centra el sidebar */
                  width: 400px; /* Ancho del sidebar */
              }
          </style>
          """, unsafe_allow_html=True)
        st.sidebar.markdown("""
          <div style="display: flex; flex-direction: column; align-items: center;">
              <h2>Procesamiento de Facturas</h2>
        """, unsafe_allow_html=True)
        st.sidebar.image("https://miro.medium.com/v2/resize:fit:4800/format:webp/1*fRYU2gXNPg8Q5wq8LLmQVw.png", use_column_width=True)
        # Mostrar imágenes
        for i, imagen in enumerate(imagenes):
            st.sidebar.image(imagen, caption=f"Página {i+1}", use_column_width=True)
            
        # Cierra el div del sidebar
        st.sidebar.markdown("</div>", unsafe_allow_html=True)

        #Intenta abrir el archivo con el mismo nombre pero txt que hay en el git si lo encuentra lo usa (ya que no tiene datos sensibles)
        #Si no lo encuentra, procesa ese PDF (descomentar para realizarlo, si no, procesa directamente el PDF)
        factura = factura_a_txt(nombre_pdf_temporal)
            
        # Eliminar el archivo PDF temporal
        os.unlink(nombre_pdf_temporal)

# Llama a la función main para ejecutar la aplicación
if __name__ == "__main__":
    main()
