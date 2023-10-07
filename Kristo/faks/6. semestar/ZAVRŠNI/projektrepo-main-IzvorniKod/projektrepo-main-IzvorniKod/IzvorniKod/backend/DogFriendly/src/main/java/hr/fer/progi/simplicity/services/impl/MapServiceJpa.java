package hr.fer.progi.simplicity.services.impl;

import hr.fer.progi.simplicity.entities.*;
import hr.fer.progi.simplicity.services.BusinessService;
import hr.fer.progi.simplicity.services.LocationService;
import hr.fer.progi.simplicity.services.MapService;
import hr.fer.progi.simplicity.services.UserRatingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class MapServiceJpa implements MapService {
    @Autowired
    LocationService locationService;

    @Autowired
    BusinessService businessService;

    @Autowired
    UserRatingService userRatingService;

    @Override
    public Map<String, List<?>> getMapInfo() {
        Map<String, List<?>> map = new HashMap<>();

        //ALL LOCATIONS
        map.put("locations", locationService.listAll());

        //ALL BUSSINESSES
        List<Business> businesses = businessService.listAll();
        map.put("business", businesses);

        return map;
    }

    @Override
    public Map<String, List<?>> getMapUserInfo(User user) {
        Map<String, List<?>> map = new HashMap<>();

        //ALL BUSSINESSES
        List<Business> businesses = businessService.listAll();
        map.put("business", businesses);

        //ALL LOCATIONS
        map.put("locations", locationService.listAll());


        //ALL USER RATED LOCATIONS
        List<UserRating> userRatings = userRatingService.listAllByUser(user);
        List<Location> positiveLocations = new ArrayList<>();
        List<Location> negativeLocations = new ArrayList<>();

        for(int i=0;i<userRatings.size();i++){
            if(userRatings.get(i).getRatingType()== RatingType.POSITIVE)
                positiveLocations.add(userRatings.get(i).getLocation());
            else
                negativeLocations.add(userRatings.get(i).getLocation());
        }

        map.put("positiveRatedLocations", positiveLocations);
        map.put("negativeRatedLocations", negativeLocations);

        return map;
    }

}
