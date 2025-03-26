from django.db import models

# Create your models here.
class Participant(models.Model):
    identifier = models.CharField(max_length=10, unique=True)
    height = models.FloatField()

    def __str__(self):
        return self.identifier