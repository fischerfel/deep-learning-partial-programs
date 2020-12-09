import os

for filename in os.listdir('.'):
    if filename.endswith('.txt'):
        new_name = filename.replace('.txt', '-X-application-android-Xv0.txt')
        ##new_name = filename.replace('-X-application-android-Xv0.txt', '.txt')
        os.rename(filename, new_name)





