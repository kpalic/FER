package hr.fer.progi.simplicity.security.requests_responses;

import hr.fer.progi.simplicity.entities.LocationType;
import hr.fer.progi.simplicity.entities.RatingType;

public class AddLocationDTO {
    private Double longitude;
    private Double latitude;
    private String name;
    private LocationType type;
    private RatingType rating;

    public Double getLongitude() {
        return longitude;
    }

    public Double getLatitude() {
        return latitude;
    }

    public String getName() {
        return name;
    }

    public LocationType getType() {
        return type;
    }

    public RatingType getRating() {
        return rating;
    }
}
