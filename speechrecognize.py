# import speech_recognition as sr
#
#
# r = sr.Recognizer()
# mic = sr.Microphone()
# print("hello")
#
# while True:
#     with mic as source:
#         audio = r.listen(source)
#     words = r.recognize_google(audio)
#     print(words)
#

#
#     print("not my word i want")
import speech_recognition as sr
import os

r = sr.Recognizer()
c = 1
while (c):
    os.system("cls")
    print("\n Speak and get as Text \n")
    try:
        with sr.Microphone() as source:

            print("\n Listening... \n")
            audio_r = r.record(source, duration=5)
            words = r.recognize_google(audio_r)
            if words == "Big":
                print(" che do goc rong")

            if words == "Small":
                print(" che do goc hep")

            if words == "Left":
                print("quay trai")

            if words == "Right":
                print("quay phai")

            if words == "Back":
                print("quay phia sau")

            if words == "Front":
                print("quay phia truoc")

            if words == "Close":
                print("dong tay")

            if words == "Open":
                print("mo tay")

            if words == "Scan":
                print("che do chay tu dong")

            if words == "Look":
                print("che do chay ban tu dong")

    except Exception as e:
        print("Error ... ! ", e)
    print(" Enter any key to play again \n ' Hit Enter to Exit' ")
    c = input("")