package hr.fer.progi.simplicity.services;

import hr.fer.progi.simplicity.entities.Location;
import hr.fer.progi.simplicity.entities.LocationType;
import hr.fer.progi.simplicity.entities.RatingType;

import java.util.List;

public interface LocationService {
    List<Location> listAll();
    Location createLocation(Double longitude, Double latitude, String name, LocationType locationType);
    void addLocationRating(long id, RatingType ratingType);
    void changeLocationRating(long id, RatingType ratingType);
    void deleteLocationRating(long id, RatingType ratingType);
    Location getLocationById(long id);
    void deleteLocationById(long id);
}
