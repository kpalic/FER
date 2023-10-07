package hr.fer.progi.simplicity.services;

public interface EmailSenderService {
    String sendEmail(String toEmail, String subject, String body);
    void sendConfirmationEmail(String username, String email);
    void sendSubscriptionEmail(String email);
}
