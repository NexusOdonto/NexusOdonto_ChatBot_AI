"""Base de conocimiento clínico ampliada y estructurada para Nexus Odonto.
Cubre procedimientos, preparaciones previas, cuidados postoperatorios,
primeros auxilios dentales y preguntas frecuentes clínicas.
"""

from typing import List, Dict

CONOCIMIENTO_CLINICO_COMPLETO: List[Dict[str, str]] = [
    # ── 1. PREPARACIONES PREVIAS ──────────────────────────────────────────
    {
        "titulo": "Preparación previa para Limpieza Dental y Profilaxis",
        "categoria": "preparacion",
        "especialidad": "Odontología General",
        "contenido": (
            "Preparación para Limpieza Dental Profesional (Profilaxis):\n"
            "- No ingerir alimentos pesados 30 minutos antes de la cita.\n"
            "- Cepillarse los dientes antes de acudir a la consulta.\n"
            "- Informar al odontólogo sobre encías sangrantes, presencia de marcapasos o sensibilidad dental previa.\n"
            "- Duración estimada: 30 a 45 minutos."
        ),
    },
    {
        "titulo": "Preparación previa para Consulta y Tratamiento de Ortodoncia",
        "categoria": "preparacion",
        "especialidad": "Ortodoncia",
        "contenido": (
            "Preparación para Ortodoncia (Brackets y Alineadores Invisibles):\n"
            "- Para la primera cita de valoración no se requiere preparación especial.\n"
            "- Para la colocación de brackets: se requiere profilaxis dental previa y boca libre de caries activas.\n"
            "- Traer radiografía panorámica y cefalométrica si el especialista las solicitó previamente.\n"
            "- Comer ligero antes de la cita, ya que la colocación inicial toma entre 60 y 90 minutos."
        ),
    },
    {
        "titulo": "Preparación previa para Endodoncia (Tratamiento de Conductos)",
        "categoria": "preparacion",
        "especialidad": "Endodoncia",
        "contenido": (
            "Preparación para Tratamiento de Endodoncia (Conductos):\n"
            "- Comer antes de la consulta; el procedimiento se realiza bajo anestesia local.\n"
            "- No suspender medicamentos habituales (presión arterial, tiroides, etc.).\n"
            "- Evitar tomar analgésicos fuertes en las 4 horas previas si se va a realizar la prueba diagnóstica del dolor.\n"
            "- Informar si padece alergias a anestésicos locales (lidocaína, mepivacaína) o antibióticos."
        ),
    },
    {
        "titulo": "Preparación previa para Cirugía Oral y Extracción de Cordales",
        "categoria": "preparacion",
        "especialidad": "Cirugía Oral",
        "contenido": (
            "Preparación para Cirugía Oral y Extracción de Cordales (Muelas del Juicio):\n"
            "- Asistir con un acompañante adulto responsable.\n"
            "- Comer ligero 2 horas antes del procedimiento (no asistir en ayuno si es con anestesia local).\n"
            "- No fumar ni consumir bebidas alcohólicas durante las 24 horas previas.\n"
            "- Usar ropa cómoda de manga corta o holgada.\n"
            "- Informar si toma anticoagulantes, aspirina o si tiene enfermedades como diabetes o hipertensión."
        ),
    },
    {
        "titulo": "Preparación previa para Blanqueamiento Dental",
        "categoria": "preparacion",
        "especialidad": "Estética Dental",
        "contenido": (
            "Preparación para Blanqueamiento Dental en Consultorio:\n"
            "- Es requisito indispensable tener una profilaxis dental realizada en los últimos 30 días.\n"
            "- No tener caries activas ni enfermedad periodontal no tratada.\n"
            "- Evitar el consumo de café, té, vino tinto, chocolate o alimentos con colorantes oscuros las 24 horas antes."
        ),
    },
    {
        "titulo": "Preparación previa para Diseño de Sonrisa y Carillas",
        "categoria": "preparacion",
        "especialidad": "Estética Dental",
        "contenido": (
            "Preparación para Diseño de Sonrisa y Carillas en Resina o Cerámica:\n"
            "- Requiere valoración previa con escaneo digital y registro fotográfico.\n"
            "- La boca debe estar sana, sin caries ni inflamación gingival.\n"
            "- Acudir con tiempo disponible (las sesiones de diseño pueden durar de 2 a 3 horas)."
        ),
    },
    {
        "titulo": "Preparación previa para Implantes Dentales",
        "categoria": "preparacion",
        "especialidad": "Implantología",
        "contenido": (
            "Preparación para Cirugía de Implantes Dentales:\n"
            "- Requiere tomografía computarizada (TAC dental 3D) previa para medir altura y densidad ósea.\n"
            "- Boca en óptimo estado de higiene y sin focos infecciosos activos.\n"
            "- Realizar enjuagues con clorhexidina al 0.12% según indicación del cirujano implantólogo."
        ),
    },

    # ── 2. CUIDADOS POSTOPERATORIOS ─────────────────────────────────────────
    {
        "titulo": "Cuidados posteriores a una Extracción Dental o Cirugía de Cordales",
        "categoria": "postoperatorio",
        "especialidad": "Cirugía Oral",
        "contenido": (
            "Cuidados postoperatorios tras Extracción o Cirugía Dental:\n"
            "1. Mantener la gasa mordida firmemente durante 30 a 45 minutos.\n"
            "2. Aplicar compresas frías o hielo envuelto en paño sobre la mejilla (intervalos de 15 minutos) las primeras 24 horas.\n"
            "3. NO escupir, NO enjuagarse con fuerza y NUNCA usar pitillo/pajilla para proteger el coágulo sanguíneo y evitar alveolitis seca.\n"
            "4. Dieta blanda y fría durante 48 horas (helados, gelatinas, yogures, purés tibios).\n"
            "5. Reposo relativo: no hacer ejercicio ni exponerse al sol durante 72 horas.\n"
            "6. Tomar los medicamentos formulados a las horas indicadas."
        ),
    },
    {
        "titulo": "Cuidados y Mantenimiento con Brackets de Ortodoncia",
        "categoria": "postoperatorio",
        "especialidad": "Ortodoncia",
        "contenido": (
            "Cuidados de Ortodoncia con Brackets:\n"
            "- Cepillarse después de cada comida usando cepillo ortodóntico y cepillos interdentales.\n"
            "- Evitar alimentos duros (hielo, frutos secos, turrones, chicharrones, morder manzanas enteras).\n"
            "- Evitar alimentos pegajosos (chicles, caramelos masticables).\n"
            "- Si un bracket se despega o un alambre causa molestia, aplicar cera de ortodoncia en la zona y solicitar cita de ajuste."
        ),
    },
    {
        "titulo": "Cuidados posteriores a un Blanqueamiento Dental (Dieta Blanca)",
        "categoria": "postoperatorio",
        "especialidad": "Estética Dental",
        "contenido": (
            "Cuidados 'Dieta Blanca' tras Blanqueamiento Dental:\n"
            "- Durante 72 horas seguir dieta blanca: evitar café, té, gaseosas oscuras, vino tinto, salsa de tomate, soya, curry y remolacha.\n"
            "- No fumar durante al menos 7 días post-tratamiento.\n"
            "- Es normal sentir una leve sensibilidad dental transitoria las primeras 24-48 horas. Se puede usar crema dental desensibilizante."
        ),
    },
    {
        "titulo": "Cuidados posteriores a Tratamiento de Conducto (Endodoncia)",
        "categoria": "postoperatorio",
        "especialidad": "Endodoncia",
        "contenido": (
            "Cuidados tras Tratamiento de Endodoncia:\n"
            "- Es normal sentir molestia o dolor leve al masticar durante 2 a 4 días; se controla con los analgésicos recetados.\n"
            "- Evitar masticar alimentos duros del lado tratado hasta colocar la restauración definitiva (corona o incrustación).\n"
            "- Si el empaste temporal se cae o presenta hinchazón visible, comunicarse de inmediato con la clínica."
        ),
    },
    {
        "titulo": "Cuidados y Uso de Retenedores de Ortodoncia",
        "categoria": "postoperatorio",
        "especialidad": "Ortodoncia",
        "contenido": (
            "Uso y cuidado de retenedores post-ortodoncia (placas Essix o Hawley):\n"
            "- Usar los retenedores según el protocolo indicado por el ortodoncista (habitualmente tiempo completo los primeros meses).\n"
            "- Lavarlos diariamente con agua fría y jabón neutro usando un cepillo suave. NUNCA usar agua caliente ya que los deforma.\n"
            "- Guardarlos siempre en su estuche cuando no estén en boca; no envolverlos en servilletas para evitar pérdidas accidentales."
        ),
    },
    {
        "titulo": "Cuidados tras Colocación de Resinas o Calzas Dentales Nuevas",
        "categoria": "postoperatorio",
        "especialidad": "Odontología General",
        "contenido": (
            "Cuidados con Resinas o Empastes Dentales Nuevos:\n"
            "- Se puede comer una vez haya pasado el efecto de la anestesia local para evitar morderse accidentalmente mejillas o labios.\n"
            "- Evitar alimentos extremadamente duros o morder objetos con las restauraciones.\n"
            "- Si al morder siente el diente 'más alto' de lo normal, comuníquese con nosotros para un ajuste de oclusión rápido de 5 minutos."
        ),
    },
    {
        "titulo": "Duración de la Anestesia Dental y Prevención de Lesiones",
        "categoria": "postoperatorio",
        "especialidad": "Odontología General",
        "contenido": (
            "Efecto de la Anestesia Dental Local:\n"
            "- El adormecimiento en labios, mejillas y lengua suele durar entre 2 y 4 horas tras la consulta.\n"
            "- Se debe evitar masticar alimentos sólidos mientras persista el adormecimiento para no causarse mordeduras graves involuntarias.\n"
            "- En niños, vigilar cuidadosamente que no se pellizquen ni muerdan los labios adormecidos."
        ),
    },

    # ── 3. PREGUNTAS FRECUENTES Y CASOS DE ATENCIÓN ───────────────────────
    {
        "titulo": "Manejo de la Sensibilidad Dental al frío, calor o dulce",
        "categoria": "faq_clinica",
        "especialidad": "Odontología General",
        "contenido": (
            "Sensibilidad Dental (Dientes sensibles al frío o calor):\n"
            "- Causas comunes: desgaste del esmalte, encías retraídas que exponen la dentina, caries o uso de cepillos duros.\n"
            "- Recomendaciones: cepillarse con cepillo de cerdas suaves y usar crema dental para sensibilidad (con nitrato de potasio).\n"
            "- Si el dolor es punzante y dura más de 30 segundos tras retirar el estímulo frío, puede haber afectación del nervio y requerir endodoncia."
        ),
    },
    {
        "titulo": "Sangrado de Encías y Síntomas de Gingivitis o Periodontitis",
        "categoria": "faq_clinica",
        "especialidad": "Periodoncia",
        "contenido": (
            "Sangrado e Inflamación de Encías (Gingivitis y Periodoncia):\n"
            "- Las encías sanas NUNCA sangran al cepillarse ni al usar hilo dental. El sangrado es signo inequívoco de inflamación por placa bacteriana o sarro.\n"
            "- La gingivitis es reversible con una profilaxis profesional y buena higiene en casa.\n"
            "- Si no se trata, progresa a periodontitis, causando retracción de encía, mal aliento y pérdida del soporte óseo del diente."
        ),
    },
    {
        "titulo": "Bruxismo, Apretamiento Dental Nocturno y Placa Miorrelajante",
        "categoria": "faq_clinica",
        "especialidad": "Odontología General",
        "contenido": (
            "Bruxismo (Apretar o rechinar los dientes):\n"
            "- Síntomas: dolor en la mandíbula o cuello al despertar, dolor de cabeza matutino, dientes desgastados o chasquidos al abrir la boca.\n"
            "- Tratamiento principal: confección de una placa neuromiorrelajante rígida nocturna para proteger el esmalte y relajar la articulación (ATM)."
        ),
    },
    {
        "titulo": "¿Qué hacer si se cae una calza, empaste o se parte un diente?",
        "categoria": "faq_clinica",
        "especialidad": "Odontología General",
        "contenido": (
            "Caída de una Resina o Diente Fracturado Leve:\n"
            "- Mantener la zona limpia mediante enjuagues suaves con agua tibia.\n"
            "- Evitar masticar alimentos duros o con temperaturas extremas en ese lado.\n"
            "- Agendar una cita de valoración prioritaria para obturar nuevamente la pieza antes de que se contamine con caries o se fracture más."
        ),
    },
    {
        "titulo": "¿Qué hacer si se cae o despega una Corona o Puente Dental?",
        "categoria": "faq_clinica",
        "especialidad": "Rehabilitación Oral",
        "contenido": (
            "Corona o Puente Dental Despegado:\n"
            "- Guardar la corona cuidadosamente en un recipiente limpio y llevarla a la cita.\n"
            "- NUNCA intentar pegarla con pegamentos caseros (como Super Bonder/cianoacrilato), ya que esto daña el diente y la encía de forma irreversible.\n"
            "- Solicitar cita con el rehabilitador oral para evaluar el muñón y recementar la prótesis adecuadamente."
        ),
    },
    {
        "titulo": "Halitosis (Mal Aliento Persistente) y Causas Principales",
        "categoria": "faq_clinica",
        "especialidad": "Odontología General",
        "contenido": (
            "Causas y Tratamiento del Mal Aliento (Halitosis):\n"
            "- Más del 85% de los casos se originan en la boca por acumulación de bacterias en el dorso de la lengua o sarro bajo las encías.\n"
            "- Medidas efectivas: uso diario de limpiador lingual, hilo dental, profilaxis periódica y abundante hidratación."
        ),
    },
    {
        "titulo": "Odontopediatría y Primera Visita Dental Infantil",
        "categoria": "faq_clinica",
        "especialidad": "Odontopediatría",
        "contenido": (
            "Atención odontológica en niños (Odontopediatría):\n"
            "- Primera cita recomendada al salir el primer diente de leche (alrededor de los 6 a 12 meses) o al cumplir 1 año de edad.\n"
            "- Tratamientos preventivos: aplicación de barniz de flúor para remineralizar, sellantes de fisuras y educación lúdica en cepillado."
        ),
    },
    {
        "titulo": "Frecuencia recomendada de Limpiezas Dentales",
        "categoria": "faq_clinica",
        "especialidad": "Odontología General",
        "contenido": (
            "Frecuencia de Limpiezas Profesionales (Profilaxis):\n"
            "- Pacientes sanos: cada 6 meses.\n"
            "- Pacientes con ortodoncia (brackets), implantes o antecedentes periodontales: cada 3 a 4 meses para evitar manchas y sarro subgingival."
        ),
    },

    # ── 4. POLÍTICAS INSTITUCIONALES NEXUS ODONTO ─────────────────────────
    {
        "titulo": "Horarios de Atención, Ubicación y Canales de Contacto Nexus Odonto",
        "categoria": "institucional",
        "especialidad": "Información General",
        "contenido": (
            "Información General de Nexus Odonto:\n"
            "- Consultorio: Nexus Odonto Centro Odontológico.\n"
            "- Dirección: Calle 100 # 15-20, Centro Médico Odontológico.\n"
            "- Horario: Lunes a Viernes de 8:00 AM a 12:00 PM y de 2:00 PM a 5:00 PM. Sábados de 8:00 AM a 12:00 PM. Domingos y festivos cerrado.\n"
            "- Receso médico: 12:00 PM a 2:00 PM (no se programan citas).\n"
            "- WhatsApp y Teléfono: +57 324 6030217.\n"
            "- Correo: soporte@nexusodonto.com."
        ),
    },
    {
        "titulo": "Políticas de Cancelación, Reprogramación y Puntualidad",
        "categoria": "politicas",
        "especialidad": "Información General",
        "contenido": (
            "Políticas de Citas Nexus Odonto:\n"
            "- Modificaciones y cancelaciones: Solicitar con anticipación a través del chat o vía telefónica.\n"
            "- Puntualidad: Llegar 10 a 15 minutos antes de la hora acordada para su comodidad.\n"
            "- Intervalos de agenda: Las citas se programan en bloques exactos de 30 minutos."
        ),
    },
    {
        "titulo": "Atención de Dolor Agudo Dental y Urgencias sin Cita Previa (Sobrecupo)",
        "categoria": "urgencias_y_dolor",
        "especialidad": "Odontología General y Urgencias",
        "contenido": (
            "Protocolo de Atención ante Dolor Agudo sin Disponibilidad en Agenda:\n"
            "- Empatía y Validación: Un dolor dental agudo es una prioridad clínica humana. Ningún paciente debe aguantarse el dolor por falta de citas en el calendario regular.\n"
            "- Atención Prioritaria por Sobrecupo: Si la agenda del día está completa, el paciente con dolor puede acudir directamente a la sede de Nexus Odonto (Calle 100 # 15-20) para ser valorado entre turnos por el odontólogo de turno para alivio del dolor y estabilización dental.\n"
            "- Medidas de Alivio Inmediato en Casa (mientras acude a la clínica):\n"
            "  1. Aplicar compresas frías sobre la mejilla externa (10 minutos con descanso) para desinflamar.\n"
            "  2. Enjuagues suaves con agua tibia y media cucharadita de sal.\n"
            "  3. NUNCA colocar aspirinas, alcohol ni remedios caseros abrasivos directamente sobre el diente o la encía, ya que causan quemaduras químicas severas.\n"
            "  4. Si no tiene alergias ni contraindicaciones, puede tomar un analgésico de venta libre habitual como medida paliativa temporal mientras es atendido por el doctor.\n"
            "- Canales Inmediatos: Línea directa de urgencias +57 324 6030217 o solicitar atención humana en el chat."
        ),
    },
]
