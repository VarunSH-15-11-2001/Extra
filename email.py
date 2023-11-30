import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

sender_email = "gptkeregodiswatching@gmail.com"
receiver_email = "varunhoskere@gmail.com"
password = "Gpt4kere!godiswatching@"  # Be cautious with your password
smtp_server = "smtp.gmail.com"
port = 587  # For starttls

message = MIMEMultipart()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = "Your email subject"
body = "your mom"
message.attach(MIMEText(body, "plain"))

try:
    server = smtplib.SMTP(smtp_server, port)
    server.starttls()  # Secure the connection
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message.as_string())
except Exception as e:
    print(e)
finally:
    server.quit()
