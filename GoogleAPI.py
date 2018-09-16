# Imports the Google Cloud client library
from google.cloud import translate
from google.cloud import texttospeech



GOOGLE_APPLICATION_CREDENTIALS = '/Users/tiarasykes/Desktop/HackMIT-7a1d1e6f29a1.json'

weatherdata = 'Weather Hazard Right Now. Please Evacute'
user_language = 'es'

def translateinfo(text, target = user_language):
    text = weatherdata
    translate_client = translate.Client()
    result = translate_client.translate(weatherdata, target_language=target)
    translation = result['translatedText']

    return translation

def audioinfo(text):
    audio_client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.types.SynthesisInput(
    text = translateinfo(weatherdata))

    voice = texttospeech.types.VoiceSelectionParams(
    language_code= user_language,
    ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)

    audio_config = texttospeech.types.AudioConfig(
    audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16)

    response = audio_client.synthesize_speech(synthesis_input, voice, audio_config)

    with open('output.wav', 'wb') as out:
        out.write(response.audio_content)


print(translateinfo(weatherdata))
file = audioinfo(translateinfo(weatherdata))
