import imaplib
import email
import codecs
mail = imaplib.IMAP4_SSL('imap.gmail.com')
mail.login('stanfordosa@gmail.com', 'ginzton')  #user / password
mail.list()
mail.select("2011? photo contest") # connect to folder with matching label

result, data = mail.uid('search', None, "ALL") # search and return uids instead
i = len(data[0].split())

for x in range(i):
    latest_email_uid = data[0].split()[x]
    result, email_data = mail.uid('fetch', latest_email_uid, '(RFC822)')
    raw_email = email_data[0][1]
    email_message = str(str(email.message_from_string(raw_email)).decode("quoted-printable"))
    save_string = str("C:\\\emaildump\\" + str(x) + ".eml") #set to   save location
    myfile = open(save_string, 'a')
    myfile.write(email_message)
    myfile.close()
