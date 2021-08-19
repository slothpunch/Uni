from django.db import models


class Document(models.Model):
    docfile = models.FileField(upload_to='models/speaker_recognition/16000_pcm_speeches/audio/random')
