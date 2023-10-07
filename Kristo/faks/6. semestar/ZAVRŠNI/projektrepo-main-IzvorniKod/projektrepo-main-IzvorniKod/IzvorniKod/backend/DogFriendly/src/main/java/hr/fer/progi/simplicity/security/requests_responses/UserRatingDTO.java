package hr.fer.progi.simplicity.security.requests_responses;

import hr.fer.progi.simplicity.entities.RatingType;

public class UserRatingDTO {

    private long ratingId;
    private String locationName;
    private RatingType ratingType;

    public UserRatingDTO(){
    }

    public UserRatingDTO(long ratingId, String locationName, RatingType ratingType) {
        this.ratingId = ratingId;
        this.locationName = locationName;
        this.ratingType = ratingType;
    }

    public long getRatingId() {
        return ratingId;
    }

    public void setRatingId(long ratingId) {
        this.ratingId = ratingId;
    }

    public String getLocationName() {
        return locationName;
    }

    public void setLocationName(String locationName) {
        this.locationName = locationName;
    }

    public RatingType getRatingType() {
        return ratingType;
    }

    public void setRatingType(RatingType ratingType) {
        this.ratingType = ratingType;
    }
}
