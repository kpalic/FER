package hr.fer.progi.simplicity.services.impl;

import hr.fer.progi.simplicity.entities.Location;
import hr.fer.progi.simplicity.entities.LocationType;
import hr.fer.progi.simplicity.entities.RatingType;
import hr.fer.progi.simplicity.repositories.LocationRepository;
import hr.fer.progi.simplicity.security.exceptions.RequestDeniedException;
import hr.fer.progi.simplicity.services.LocationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.util.Assert;

import java.util.List;

@Service
public class LocationServiceJpa implements LocationService {

    @Autowired
    private LocationRepository locationRepository;

    @Override
    public List<Location> listAll() {
        return locationRepository.findAll();
    }

    @Override
    public Location createLocation(Double longitude, Double latitude, String name, LocationType locationType) {
        Assert.notNull(name, "Naziv lokacije mora biti predano.");
        Assert.hasText(name, "Naziv lokacije mora biti postavljeno.");
        Assert.notNull(locationType, "Tip lokacije mora biti predan.");
        Location location = locationRepository.findByName(name);
        if(location!=null) throw new RequestDeniedException("Lokacija s imenom '" + name + "' već postoji.");

        return locationRepository.save(new Location(longitude, latitude, name, locationType));
    }

    @Override
    public void addLocationRating(long id, RatingType ratingType) {
        Location location = getLocationById(id);

        location.setVotesSum(location.getVotesSum()+1);

        if(ratingType==RatingType.POSITIVE)
            location.setPositiveVotes(location.getPositiveVotes()+1);

        location.setRating((double)location.getPositiveVotes()/(double) location.getVotesSum());
        locationRepository.save(location);
    }

    @Override
    public void deleteLocationRating(long id, RatingType ratingType) {
        Location location = getLocationById(id);

        location.setVotesSum(location.getVotesSum() - 1);

        if (location.getVotesSum() == 0) {
            deleteLocationById(id);
            return;
        }

        if (ratingType == RatingType.POSITIVE){
            location.setPositiveVotes(location.getPositiveVotes() - 1);
        }

        location.setRating((double)location.getPositiveVotes()/(double) location.getVotesSum());
        locationRepository.save(location);
    }

    @Override
    public void changeLocationRating(long id, RatingType ratingType) {
        Location location = getLocationById(id);
        if(ratingType==RatingType.POSITIVE) {
            location.setPositiveVotes(location.getPositiveVotes()+1);
        } else {
            location.setPositiveVotes(location.getPositiveVotes()-1);
        }

        location.setRating((double)location.getPositiveVotes()/(double) location.getVotesSum());
        locationRepository.save(location);
    }

    @Override
    public Location getLocationById(long id) {
        return locationRepository.findById(id);
    }

    @Override
    public void deleteLocationById(long id) {
        locationRepository.deleteById(id);
    }
}
