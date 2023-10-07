package hr.fer.progi.simplicity.security.requests_responses;

public class PromoteBusinessDTO {

    private String businessOIB;
    private String promoteDuration;

    public PromoteBusinessDTO(){
    }

    public PromoteBusinessDTO(String businessOIB, String promoteDuration) {
        this.businessOIB = businessOIB;
        this.promoteDuration = promoteDuration;
    }

    public String getBusinessOIB() {
        return businessOIB;
    }

    public String getPromoteDuration() {
        return promoteDuration;
    }
}
