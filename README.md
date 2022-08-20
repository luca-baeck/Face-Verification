<h1 align="center">Face Verification</h1>

<p align="center">Bau einer Application, die ein deep learning Model erstellt, mit dem man eine
Person über eine Webcam in Echtzeit erkennen, identifizieren und verifizieren kann.</p>

## Links

- [Repository](https://github.com/luca-baeck/Face-Verification "Face Verification Repository")

- [Bugs](https://github.com/Rohit19060/Face-Verification/issues "Issues Page")


## Screenshots

![Screenshot](/application-preview/verified.png "Screenshot")
![Screenshot](/application-preview/unverified.png "Screenshot")
![Screenshot](/application-preview/trained.png "Screenshot")

![](/screenshots/2.png)

![](/screenshots/3.png)

## Available Commands

Im Programm kannst du folgende Befehle ausführen:

### `Train Siamese Network`,

Dies wird das neuronale Netz mit den zuvor aufgenommenen Testdaten von dir trainieren. In der Konfigurationsdatei kannst du die Anzahl der Trainingsepochen festlegen.

### `Test Siamese Network`,

Dies wird das neuronale Netz mit einem zufälligen Datentestpacket testen und so aktuelle Werte wie den Recall und die Precision deines Models liefern.

### `Set New Face`,

Dies wird alle bisherigen Daten löschen und über die Webcam ein neues Gesicht als positives Bild hinterlegen.

### `Verify`,

Dies wird ein Bild über die Webcam aufnehmen und dem Model übergeben, welches dann bestimmt ob es sich in dem Bild um die zu verifizierende Person handelt.

## Sprachen

- Python

## IDE

- PyCharm

## Autor

**Luca Bäck**

- [Profil](https://github.com/luca-baeck "Luca Bäck")
- [Email](mailto:luca.baeck@outlook.de?subject=Hello "Hi!")
