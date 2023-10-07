package hr.fer.progi.simplicity.services.impl;

import hr.fer.progi.simplicity.services.EmailSenderService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

@Service
public class EmailSenderServiceJpa implements EmailSenderService {

    @Autowired
    private JavaMailSender mailSender;

    @Override
    public String sendEmail(String toEmail, String subject, String body) {
        SimpleMailMessage message = new SimpleMailMessage();
        message.setFrom("noreply.dogfriendly@gmail.com");
        message.setTo(toEmail);
        message.setText(body);
        message.setSubject(subject);

        mailSender.send(message);

        return "Mail sent successfully!";
    }

    @Override
    public void sendConfirmationEmail(String username, String email) {
        this.sendEmail(email, "Dog friendly potvrda registracije",
                "Pozdrav, molimo Vas da klikom na link potvrdite svoju registraciju na web stranici DogFriendly:\n\nhttps://dogfriendly-frontend.onrender.com/auth/email-confirm/?username=" + username);

    }

    @Override
    public void sendSubscriptionEmail(String email) {
        this.sendEmail(email, "Dog friendly potvrda plaćanja",
                "Čestitamo, uspješno ste platili svoju mjesečnu pretplatu!");
    }
}
