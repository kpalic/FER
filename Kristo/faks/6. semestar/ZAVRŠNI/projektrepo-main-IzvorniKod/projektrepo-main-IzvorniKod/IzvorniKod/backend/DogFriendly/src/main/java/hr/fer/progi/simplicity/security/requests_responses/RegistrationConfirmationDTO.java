package hr.fer.progi.simplicity.security.requests_responses;

public class RegistrationConfirmationDTO {

    private String username;

    public RegistrationConfirmationDTO(){
    }

    public RegistrationConfirmationDTO(String username) {
        this.username = username;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }
}
