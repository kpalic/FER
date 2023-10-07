package hr.fer.progi.simplicity.security.requests_responses;

import java.time.LocalDate;

public class OwnerProfileDTO {

    private String username;
    private String email;

    private String businessName;
    private String businessType;
    private String businessAddress;
    private String businessCity;
    private String businessOIB;
    private String businessMobileNumber;
    private String businessDescription;

    private String cardNumber;

    private LocalDate promotionStart;
    private String promotionDuration;


    public OwnerProfileDTO(){
    }

    public OwnerProfileDTO(String username, String email, String businessName, String businessType, String businessAddress, String businessCity, String businessOIB, String businessMobileNumber, String businessDescription, String cardNumber, LocalDate promotionStart, String promotionDuration) {
        this.username = username;
        this.email = email;
        this.businessName = businessName;
        this.businessType = businessType;
        this.businessAddress = businessAddress;
        this.businessCity = businessCity;
        this.businessOIB = businessOIB;
        this.businessMobileNumber = businessMobileNumber;
        this.businessDescription = businessDescription;
        this.cardNumber = cardNumber;
        this.promotionStart = promotionStart;
        this.promotionDuration = promotionDuration;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getBusinessName() {
        return businessName;
    }

    public void setBusinessName(String businessName) {
        this.businessName = businessName;
    }

    public String getBusinessType() {
        return businessType;
    }

    public void setBusinessType(String businessType) {
        this.businessType = businessType;
    }

    public String getBusinessAddress() {
        return businessAddress;
    }

    public void setBusinessAddress(String businessAddress) {
        this.businessAddress = businessAddress;
    }

    public String getBusinessOIB() {
        return businessOIB;
    }

    public void setBusinessOIB(String businessOIB) {
        this.businessOIB = businessOIB;
    }

    public String getBusinessMobileNumber() {
        return businessMobileNumber;
    }

    public void setBusinessMobileNumber(String businessMobileNumber) {
        this.businessMobileNumber = businessMobileNumber;
    }

    public String getBusinessDescription() {
        return businessDescription;
    }

    public void setBusinessDescription(String businessDescription) {
        this.businessDescription = businessDescription;
    }

    public String getCardNumber() {
        return cardNumber;
    }

    public void setCardNumber(String cardNumber) {
        this.cardNumber = cardNumber;
    }

    public LocalDate getPromotionStart() {
        return promotionStart;
    }

    public void setPromotionStart(LocalDate promotionStart) {
        this.promotionStart = promotionStart;
    }

    public String getPromotionDuration() {
        return promotionDuration;
    }

    public void setPromotionDuration(String promotionDuration) {
        this.promotionDuration = promotionDuration;
    }

    public String getBusinessCity() {
        return businessCity;
    }

    public void setBusinessCity(String businessCity) {
        this.businessCity = businessCity;
    }
}
