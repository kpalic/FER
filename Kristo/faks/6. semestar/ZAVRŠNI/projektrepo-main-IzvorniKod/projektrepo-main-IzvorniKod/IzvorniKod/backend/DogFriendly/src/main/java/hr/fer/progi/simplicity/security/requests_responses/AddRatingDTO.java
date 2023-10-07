package hr.fer.progi.simplicity.security.requests_responses;

import hr.fer.progi.simplicity.entities.RatingType;

public class AddRatingDTO {
    private long id;
    private RatingType rating;

    public long getId() {
        return id;
    }

    public RatingType getRating() {
        return rating;
    }
}
