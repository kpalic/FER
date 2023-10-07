package hr.fer.progi.simplicity.security.requests_responses;

public class UpdateDataDTO {

    private String username;
    private String password;

    private String businessName;
    private String businessDescription;

    private String ratingId;
    private String ratingType;

    public UpdateDataDTO(){
    }

    public UpdateDataDTO(String username, String password, String businessName, String businessDescription, String ratingId, String ratingType) {
        this.username = username;
        this.password = password;
        this.businessName = businessName;
        this.businessDescription = businessDescription;
        this.ratingId = ratingId;
        this.ratingType = ratingType;
    }

    public String getUsername() {
        return username;
    }

    public String getPassword() {
        return password;
    }

    public String getBusinessName() {
        return businessName;
    }

    public String getBusinessDescription() {
        return businessDescription;
    }

    public String getRatingId() {
        return ratingId;
    }

    public String getRatingType() {
        return ratingType;
    }
}
