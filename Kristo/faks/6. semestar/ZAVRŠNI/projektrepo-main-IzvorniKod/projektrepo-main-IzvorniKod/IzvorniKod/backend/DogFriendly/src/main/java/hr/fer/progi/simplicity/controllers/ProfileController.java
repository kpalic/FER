package hr.fer.progi.simplicity.controllers;

import hr.fer.progi.simplicity.entities.*;
import hr.fer.progi.simplicity.security.CustomUserDetailsService;
import hr.fer.progi.simplicity.security.exceptions.EntityMissingException;
import hr.fer.progi.simplicity.security.exceptions.RequestDeniedException;
import hr.fer.progi.simplicity.security.jwt.JwtAuthenticationFilter;
import hr.fer.progi.simplicity.security.jwt.JwtTokenProvider;
import hr.fer.progi.simplicity.security.requests_responses.*;
import hr.fer.progi.simplicity.services.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import javax.servlet.http.HttpServletRequest;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;
import java.util.List;

//@CrossOrigin(origins="https://dogfriendly-frontservice.onrender.com")
@CrossOrigin(origins="https://dogfriendly-frontend.onrender.com")

// @CrossOrigin(origins="http://localhost:3000")
//@CrossOrigin(origins="https://dogfriendly-frontservice.onrender.com")
@RestController
@RequestMapping("/profile")
public class ProfileController {

    @Autowired
    JwtTokenProvider tokenProvider;

    @Autowired
    ProfileService profileService;

    @Autowired
    CustomUserDetailsService customUserDetailsService;

    @Autowired
    BusinessService businessService;

    @Autowired
    CardService cardService;

    @Autowired
    UserRatingService userRatingService;

    @Value("${app.encodingKey}")
    private String encodingKey;

    @Value("${app.ivVector}")
    private String ivVector;

    @GetMapping("/user")
    public UserProfileDTO showUserProfile(HttpServletRequest request){
        JwtAuthenticationFilter filter = new JwtAuthenticationFilter();
        String token = filter.getJwtFromRequest(request);
        Long id = tokenProvider.getUserIdFromJWT(token);
        User userDB = customUserDetailsService.getUserById(id).orElseThrow(
                () -> new EntityMissingException(User.class, id));

        Comparator<UserRating> comparator = new Comparator<UserRating>() {
            @Override
            public int compare(UserRating o1, UserRating o2) {
                String l1Name = o1.getLocation().getName();
                String l2Name = o2.getLocation().getName();

                if (l1Name.compareTo(l2Name) > 0){
                    return 1;
                }else if (l1Name.compareTo(l2Name) < 0){
                    return -1;
                }else{
                    return 0;
                }
            }
        };

        List<UserRating> ratingsDB = userRatingService.listAllByUser(userDB);
        ratingsDB.sort(comparator);
        List<UserRatingDTO> ratings = new ArrayList<>();

        for (int i = 0; i < ratingsDB.size(); i++){
            UserRating ratingDB = ratingsDB.get(i);
            UserRatingDTO rating = new UserRatingDTO(ratingDB.getId(), ratingDB.getLocation().getName(), ratingDB.getRatingType());
            ratings.add(rating);
        }

        UserProfileDTO userProfile = new UserProfileDTO(userDB.getUsername(), userDB.getEmail(), ratings);

        return userProfile;
    }

    @GetMapping("/owner")
    public OwnerProfileDTO showOwnerProfile(HttpServletRequest request){
        JwtAuthenticationFilter filter = new JwtAuthenticationFilter();
        String token = filter.getJwtFromRequest(request);
        User userDB = customUserDetailsService.getUserById(tokenProvider.getUserIdFromJWT(token)).orElseThrow(
                () -> new EntityMissingException(User.class, tokenProvider.getUserIdFromJWT(token)));
        
       try {
           Business businessDB = ((Owner) userDB).getBusiness();

           Card cardDB = ((Owner) userDB).getCard();
           String cardNumber = cardService.decrypt(cardDB.getCardNumber(), encodingKey, ivVector);

           String cardNumbers = cardNumber.substring(0, 4);
           OwnerProfileDTO ownerProfile = new OwnerProfileDTO(userDB.getUsername(), userDB.getEmail(),
                   businessDB.getBusinessName(), businessDB.getBusinessType().toString(), businessDB.getBusinessAddress(), businessDB.getBusinessCity(), businessDB.getBusinessOIB(),
                   businessDB.getBusinessMobileNumber(), businessDB.getBusinessDescription(),
                   cardNumbers, businessDB.getPromotionStart(), businessDB.getPromotionDuration());

           if (ownerProfile.getPromotionDuration() != null && ownerProfile.getPromotionStart() != null){
               Date currentDate = new Date();
               LocalDate startDate;
               if (ownerProfile.getPromotionDuration().equals("0.25")) {
            	   startDate = ownerProfile.getPromotionStart().plusDays(7);
               } else {            	   
            	   startDate = ownerProfile.getPromotionStart().plusMonths(Long.parseLong(ownerProfile.getPromotionDuration()));
               }
               
               Date checkDate = java.sql.Date.valueOf(startDate);
               if (checkDate.getTime() < currentDate.getTime()){
                   ownerProfile.setPromotionStart(null);
                   ownerProfile.setPromotionDuration(null);
               }
           }
           return ownerProfile;

       } catch (Exception e) {
           throw new RequestDeniedException("Greska u showOwnerProfile");
       }

    }

    @PutMapping("/user/edit")
    public ResponseEntity<?> updateUser(HttpServletRequest request, @RequestBody UpdateDataDTO updateDataDTO){
        JwtAuthenticationFilter filter = new JwtAuthenticationFilter();
        String token = filter.getJwtFromRequest(request);
        Long id = tokenProvider.getUserIdFromJWT(token);
        User userDB= customUserDetailsService.getUserById(id).orElseThrow(
                () -> new EntityMissingException(User.class, id));


        String newUsername = updateDataDTO.getUsername();
        String newPassword = updateDataDTO.getPassword();

        try{
            if ((!newUsername.equals("")) && userDB.getUsername().equals(newUsername) == false){
                profileService.updateUsername(userDB.getId(), newUsername);
            }

            if (!newPassword.equals("")){
                profileService.updatePassword(userDB.getId(), newPassword);
            }
        } catch (Exception e) {
            throw new RequestDeniedException(e.getMessage());
        }

        return new ResponseEntity(new ApiResponseDTO(true, "Data update completed successfully!"),
                HttpStatus.CREATED);
    }

    @PutMapping("/owner/edit")
    public ResponseEntity<?> updateOwner(HttpServletRequest request, @RequestBody UpdateDataDTO updateDataDTO) {
        JwtAuthenticationFilter filter = new JwtAuthenticationFilter();
        String token = filter.getJwtFromRequest(request);
        Long id = tokenProvider.getUserIdFromJWT(token);
        User userDB = customUserDetailsService.getUserById(id).orElseThrow(
                () -> new EntityMissingException(User.class, id));

        String newUsername = updateDataDTO.getUsername();
        String newPassword = updateDataDTO.getPassword();
        String newBusinessName = updateDataDTO.getBusinessName();
        String newBusinessDescription = updateDataDTO.getBusinessDescription();

        try{
            if ((!newUsername.equals("")) && userDB.getUsername().equals(newUsername) == false) {
                profileService.updateUsername(userDB.getId(), newUsername);
            }

            if (!newPassword.equals("")) {
                profileService.updatePassword(userDB.getId(), newPassword);
            }


            String businessNameDB = ((Owner) userDB).getBusiness().getBusinessName();
            if ((!newBusinessName.equals("")) && businessNameDB.equals(newBusinessName) == false) {
                businessService.updateBusinessName(((Owner) userDB).getBusiness().getId(), newBusinessName);
            }

            if (!newBusinessDescription.equals("")) {
                businessService.updateBusinessDescription(((Owner) userDB).getBusiness().getId(), newBusinessDescription);
            }
        } catch (Exception e) {
            throw new RequestDeniedException(e.getMessage());
        }



        return new ResponseEntity(new ApiResponseDTO(true, "Data update completed successfully!"),
                HttpStatus.CREATED);
    }

    @PutMapping("/user/editRating")
    public ResponseEntity<?> updateUserRating(@RequestBody UpdateDataDTO updateDataDTO){
        long ratingId = Long.parseLong(updateDataDTO.getRatingId());
        RatingType ratingType = RatingType.POSITIVE;
        if (updateDataDTO.getRatingType().equals("NEGATIVE")){
            ratingType = RatingType.NEGATIVE;
        }

        try{
            userRatingService.changeRatingForUser(ratingId, ratingType);
        } catch (Exception e) {
            throw new RequestDeniedException(e.getMessage());
        }

        return new ResponseEntity(new ApiResponseDTO(true, "Data update completed successfully!"),
                HttpStatus.CREATED);
    }

    @DeleteMapping("/user/ratingDelete")
    public ResponseEntity<?> deleteRating(@RequestParam String ratingId){
        long id = Long.parseLong(ratingId);

        try{
            userRatingService.deleteRatingForUser(id);;
        } catch (Exception e) {
            throw new RequestDeniedException(e.getMessage());
        }

        return new ResponseEntity(new ApiResponseDTO(true, "Rating deleted successfully!"),
                HttpStatus.CREATED);
    }

    @DeleteMapping("/user/delete")
    public ResponseEntity<?> deleteUser(HttpServletRequest request){
        JwtAuthenticationFilter filter = new JwtAuthenticationFilter();
        String token = filter.getJwtFromRequest(request);
        Long id = tokenProvider.getUserIdFromJWT(token);
        User userDB = customUserDetailsService.getUserById(id).orElseThrow(
                () -> new EntityMissingException(User.class, id));

        try{
            profileService.deleteUserById(userDB.getId());
        } catch (Exception e) {
            throw new RequestDeniedException(e.getMessage());
        }

        return new ResponseEntity(new ApiResponseDTO(true, "User deleted successfully!"),
                HttpStatus.CREATED);
    }

    @DeleteMapping("/owner/delete")
    public ResponseEntity<?> deleteOwner(HttpServletRequest request){
        JwtAuthenticationFilter filter = new JwtAuthenticationFilter();
        String token = filter.getJwtFromRequest(request);
        Long id = tokenProvider.getUserIdFromJWT(token);
        User userDB = customUserDetailsService.getUserById(id).orElseThrow(
                () -> new EntityMissingException(User.class, id));

        try{
            profileService.deleteUserById(userDB.getId());
        } catch (Exception e) {
            throw new RequestDeniedException(e.getMessage());
        }

        return new ResponseEntity(new ApiResponseDTO(true, "User deleted successfully!"),
                HttpStatus.CREATED);
    }

    @PostMapping("/owner/promote")
    public ResponseEntity<?> promoteBusiness(@RequestBody PromoteBusinessDTO promoteBusinessDTO){
        try{
            businessService.setPromoteDuration(promoteBusinessDTO.getBusinessOIB(), promoteBusinessDTO.getPromoteDuration());
        } catch (Exception e) {
            throw new RequestDeniedException(e.getMessage());
        }

        return new ResponseEntity(new ApiResponseDTO(true, "Promotion successfully set up!"),
                HttpStatus.CREATED);
    }
}
