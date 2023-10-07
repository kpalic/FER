package hr.fer.progi.simplicity.entities;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import javax.validation.constraints.Size;
import java.time.LocalDate;

@Entity
public class Business {
    @Id
    @GeneratedValue
    private Long id;

    @Column(unique = true)
    @NotNull
    private String businessName;

    @NotNull
    private BusinessType businessType;

    @NotNull
    private String businessAddress;

    @Column(nullable = false)
    @NotNull
    private String businessCity;

    @Column(unique = true, nullable = false)
    @NotNull
    @Size(min=11, max=11)
    private String businessOIB;

    @NotNull
    private String businessMobileNumber;
    @NotNull
    private String businessDescription;

    private LocalDate promotionStart;
    private String promotionDuration;

    public Business() {
    }

    public Business(String businessName, BusinessType businessType, String businessAddress, String businessCity, String businessOIB, String businessMobileNumber, String businessDescription) {
        this.businessName = businessName;
        this.businessType = businessType;
        this.businessAddress = businessAddress;
        this.businessCity = businessCity;
        this.businessOIB = businessOIB;
        this.businessMobileNumber = businessMobileNumber;
        this.businessDescription = businessDescription;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getBusinessName() {
        return businessName;
    }

    public void setBusinessName(String businessName) {
        this.businessName = businessName;
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

    public String getBusinessDescription() {
        return businessDescription;
    }

    public void setBusinessDescription(String businessDescription) {
        this.businessDescription = businessDescription;
    }

    public String getBusinessMobileNumber() {
        return businessMobileNumber;
    }

    public void setBusinessMobileNumber(String businessMobileNumber) {
        this.businessMobileNumber = businessMobileNumber;
    }

    public BusinessType getBusinessType() {
        return businessType;
    }

    public void setBusinessType(BusinessType businessType) {
        this.businessType = businessType;
    }

    public String getPromotionDuration() {
        return promotionDuration;
    }

    public void setPromotionDuration(String promoteDuration) {
        this.promotionDuration = promoteDuration;
    }

    public String getBusinessCity() {
        return businessCity;
    }

    public void setBusinessCity(String businessCity) {
        this.businessCity = businessCity;
    }

    public LocalDate getPromotionStart() {
        return promotionStart;
    }

    public void setPromotionStart(LocalDate promoteStart) {
        this.promotionStart = promoteStart;
    }
}
