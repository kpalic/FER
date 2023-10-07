package hr.fer.progi.simplicity.services;

import hr.fer.progi.simplicity.entities.Location;
import hr.fer.progi.simplicity.entities.RatingType;
import hr.fer.progi.simplicity.entities.User;
import hr.fer.progi.simplicity.entities.UserRating;

import java.util.List;

public interface UserRatingService {
    List<UserRating> listAllByUser(User user);
    List<UserRating> listAllByLocation(Location location);
    UserRating addRatingForUser(User user, Location location, RatingType ratingType);
    UserRating changeRatingForUser (long id, RatingType ratingType);
    void deleteRatingForUser (long id);
    void deleteAllByUser(User user);
}
