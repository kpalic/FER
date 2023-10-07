package hr.fer.progi.simplicity.controllers;

import hr.fer.progi.simplicity.entities.*;
import hr.fer.progi.simplicity.security.CustomUserDetailsService;
import hr.fer.progi.simplicity.security.exceptions.EntityMissingException;
import hr.fer.progi.simplicity.security.jwt.JwtAuthenticationFilter;
import hr.fer.progi.simplicity.security.jwt.JwtTokenProvider;
import hr.fer.progi.simplicity.security.requests_responses.AddLocationDTO;
import hr.fer.progi.simplicity.security.requests_responses.AddRatingDTO;
import hr.fer.progi.simplicity.security.requests_responses.ApiResponseDTO;
import hr.fer.progi.simplicity.services.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;
import java.util.List;
import java.util.Map;

@CrossOrigin(origins="https://dogfriendly-frontend.onrender.com")
//@CrossOrigin(origins="http://localhost:3000")
//@CrossOrigin(origins="https://dogfriendly-frontservice.onrender.com")
@RestController
@RequestMapping("/map")
public class MapController {
    @Autowired
    MapService mapService;

    @Autowired
    CustomUserDetailsService userService;

    @Autowired
    LocationService locationService;

    @Autowired
    UserRatingService userRatingService;

    @Autowired
    JwtTokenProvider tokenProvider;

    @GetMapping("")
    public Map<String, List<?>> mapInfo(HttpServletRequest request) {
        JwtAuthenticationFilter filter = new JwtAuthenticationFilter();
        String token = filter.getJwtFromRequest(request);
        if (token == null) {
            return mapService.getMapInfo();
        } else {
            Long id = tokenProvider.getUserIdFromJWT(token);
            User user = userService.getUserById(id).orElseThrow(
                    () -> new EntityMissingException(User.class, id));

            if(user.getRole()== RoleType.USER) {
                return mapService.getMapUserInfo(user);
            } else {
                return mapService.getMapInfo();
            }
        }
    }

    @PostMapping("")
    @PreAuthorize("hasAuthority('USER')")
    public ResponseEntity<?> addLocationOnMap(HttpServletRequest request, @RequestBody AddLocationDTO addLocationDTO) {
        JwtAuthenticationFilter filter = new JwtAuthenticationFilter();
        String token = filter.getJwtFromRequest(request);
        Long id = tokenProvider.getUserIdFromJWT(token);
        User user = userService.getUserById(id).orElseThrow(
                () -> new EntityMissingException(User.class, id));

        Location location = locationService.createLocation(addLocationDTO.getLongitude(),
                                                           addLocationDTO.getLatitude(),
                                                           addLocationDTO.getName(),
                                                           addLocationDTO.getType());
        userRatingService.addRatingForUser(user, location, addLocationDTO.getRating());

        return new ResponseEntity(new ApiResponseDTO(true, "Location created successfully!"),
                                                  HttpStatus.CREATED);
    }

    @PutMapping("")
    @PreAuthorize("hasAuthority('USER')")
    public ResponseEntity<?> addRatingToLocation(HttpServletRequest request, @RequestBody AddRatingDTO addRatingDTO) {
        JwtAuthenticationFilter filter = new JwtAuthenticationFilter();
        String token = filter.getJwtFromRequest(request);
        Long id = tokenProvider.getUserIdFromJWT(token);
        User user = userService.getUserById(id).orElseThrow(
                () -> new EntityMissingException(User.class, id));

        Location location = locationService.getLocationById(addRatingDTO.getId());
        userRatingService.addRatingForUser(user, location, addRatingDTO.getRating());

        return new ResponseEntity(new ApiResponseDTO(true, "Rating added successfully!"),
                HttpStatus.OK);
    }


}



