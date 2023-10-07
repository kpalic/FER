package hr.fer.progi.simplicity.security.requests_responses;

import java.util.Date;

public class OwnerRegistrationDTO {
    private String username;
    private String email;
    private String password;

    private String businessName;
    private String businessType;
    private String businessAddress;
    private String businessCity;
    private String businessOIB;
    private String businessMobileNumber;
    private String businessDescription;

    private String cardNumber;
    private String expiryDateMonth;
    private String expiryDateYear;
    private String cvv;

    public String getUsername() {
        return username;
    }

    public String getEmail() {
        return email;
    }

    public String getPassword() {
        return password;
    }

    public String getBusinessName() {
        return businessName;
    }

    public String getBusinessType() {
        return businessType;
    }

    public String getBusinessAddress() {
        return businessAddress;
    }

    public String getBusinessOIB() {
        return businessOIB;
    }

    public String getBusinessMobileNumber() {
        return businessMobileNumber;
    }

    public String getBusinessDescription() {
        return businessDescription;
    }

    public String getCardNumber() {
        return cardNumber;
    }

    public String getCvv() {
        return cvv;
    }

    public String getExpiryDateMonth() {
        return expiryDateMonth;
    }

    public String getExpiryDateYear() {
        return expiryDateYear;
    }

    public String getBusinessCity() {
        return businessCity;
    }
}
