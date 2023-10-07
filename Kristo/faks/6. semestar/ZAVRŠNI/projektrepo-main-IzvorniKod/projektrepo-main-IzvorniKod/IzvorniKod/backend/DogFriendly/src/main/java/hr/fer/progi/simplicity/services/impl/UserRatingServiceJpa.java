package hr.fer.progi.simplicity.services.impl;

import hr.fer.progi.simplicity.entities.Location;
import hr.fer.progi.simplicity.entities.RatingType;
import hr.fer.progi.simplicity.entities.User;
import hr.fer.progi.simplicity.entities.UserRating;
import hr.fer.progi.simplicity.repositories.UserRatingRepository;
import hr.fer.progi.simplicity.security.exceptions.RequestDeniedException;
import hr.fer.progi.simplicity.services.UserRatingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class UserRatingServiceJpa implements UserRatingService {
    @Autowired
    private UserRatingRepository userRatingRepository;

    @Autowired
    private LocationServiceJpa locationServiceJpa;

    @Override
    public List<UserRating> listAllByUser(User user) {
        return userRatingRepository.findAllByUser(user);
    }

    @Override
    public List<UserRating> listAllByLocation(Location location) {
        return userRatingRepository.findAllByLocation(location);
    }

    @Override
    public UserRating addRatingForUser(User user, Location location, RatingType ratingType) {
        List<UserRating> ratings = listAllByLocation(location);

        for (int i = 0; i < ratings.size(); i++){
            String usernameDB = ratings.get(i).getUser().getUsername();
            if (user.getUsername().equals(usernameDB)){
                throw new RequestDeniedException("Korisnik je lokaciju vec ocijenio!");
            }
        }

        locationServiceJpa.addLocationRating(location.getId(), ratingType);
        return userRatingRepository.save(new UserRating(user, location, ratingType));
    }

    @Override
    public UserRating changeRatingForUser(long id, RatingType ratingType) {
        UserRating userRating = userRatingRepository.getUserRatingById(id);
        userRating.setRatingType(ratingType);

        locationServiceJpa.changeLocationRating(userRating.getLocation().getId(), ratingType);
        return userRatingRepository.save(userRating);
    }

    @Override
    public void deleteRatingForUser(long id) {
        UserRating userRating = userRatingRepository.getUserRatingById(id);
        long locationId = userRating.getLocation().getId();
        RatingType ratingType = userRating.getRatingType();

        userRatingRepository.deleteById(id);
        locationServiceJpa.deleteLocationRating(locationId, ratingType);
    }

    @Override
    public void deleteAllByUser(User user) {
        List<UserRating> userRatings = userRatingRepository.findAllByUser(user);

        for(int i=0;i<userRatings.size();i++){
            deleteRatingForUser(userRatings.get(i).getId());
        }
    }
}
