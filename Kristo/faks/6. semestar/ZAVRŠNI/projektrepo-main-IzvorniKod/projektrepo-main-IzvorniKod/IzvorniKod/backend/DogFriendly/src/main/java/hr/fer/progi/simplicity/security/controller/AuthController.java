package hr.fer.progi.simplicity.security.controller;

import hr.fer.progi.simplicity.entities.RoleType;
import hr.fer.progi.simplicity.security.exceptions.EntityMissingException;
import hr.fer.progi.simplicity.security.requests_responses.*;
import hr.fer.progi.simplicity.entities.User;
import hr.fer.progi.simplicity.security.CustomUserDetailsService;
import hr.fer.progi.simplicity.security.jwt.JwtAuthenticationResponse;
import hr.fer.progi.simplicity.security.jwt.JwtTokenProvider;
import hr.fer.progi.simplicity.services.ProfileService;
import hr.fer.progi.simplicity.security.exceptions.RequestDeniedException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;

//@CrossOrigin(origins="https://dogfriendly-frontservice.onrender.com")
@CrossOrigin(origins="https://dogfriendly-frontend.onrender.com")
// @CrossOrigin(origins="http://localhost:3000")
//@CrossOrigin(origins="https://dogfriendly-frontservice.onrender.com")
@RestController
@RequestMapping("/auth")
public class AuthController {

    @Autowired
    AuthenticationManager authenticationManager;

    @Autowired
    CustomUserDetailsService customUserDetailsService;

    @Autowired
    BCryptPasswordEncoder passwordEncoder;

    @Autowired
    JwtTokenProvider tokenProvider;

    @Autowired
    private ProfileService profileService;


    @PostMapping("/register/user")
    public ResponseEntity<?> registerUser(@Valid @RequestBody UserRegistrationDTO userRegistrationDTO) {
        //CREATE USER
        User user = customUserDetailsService.createUser(userRegistrationDTO.getUsername(),
                                       userRegistrationDTO.getEmail(),
                                       userRegistrationDTO.getPassword(),
                                       RoleType.USER);

        return new ResponseEntity(new ApiResponseDTO(true, "User registered successfully!"),
                                  HttpStatus.CREATED);
    }

    @PostMapping("/register/owner")
    public ResponseEntity<?> registerOwner(@Valid @RequestBody OwnerRegistrationDTO ownerRegistrationDTO) {
        //CREATE USER
        User user = customUserDetailsService.createOwner(ownerRegistrationDTO.getUsername(),
                                       ownerRegistrationDTO.getEmail(),
                                       ownerRegistrationDTO.getPassword(),
                                       ownerRegistrationDTO.getBusinessName(),
                                       ownerRegistrationDTO.getBusinessType(),
                                       ownerRegistrationDTO.getBusinessAddress(),
                                       ownerRegistrationDTO.getBusinessCity(),
                                       ownerRegistrationDTO.getBusinessOIB(),
                                       ownerRegistrationDTO.getBusinessMobileNumber(),
                                       ownerRegistrationDTO.getBusinessDescription(),
                                       ownerRegistrationDTO.getCardNumber(),
                                       ownerRegistrationDTO.getExpiryDateMonth(),
                                       ownerRegistrationDTO.getExpiryDateYear(),
                                       ownerRegistrationDTO.getCvv());

        //SEND MAIL FOR SUCCESSFUL PAYMENT?????

        return new ResponseEntity(new ApiResponseDTO(true, "Owner registered successfully!"),
                                                 HttpStatus.CREATED);
    }

    @PostMapping("/email-confirm")
    public ResponseEntity<?> emailConfirmation(@RequestBody RegistrationConfirmationDTO registrationConfirmationDTO){
        User userDB = null;

        try{
            userDB = customUserDetailsService.getUserByUsername(registrationConfirmationDTO.getUsername());
        } catch (Exception e) {
            throw new RequestDeniedException("Nemoguće pronaći korisnika u bazi podataka.");
        }

        profileService.updateAccountActivatedByEmail(userDB.getId(), true);

        return new ResponseEntity(new ApiResponseDTO(true, "Successful email registration confirmation!"),
                HttpStatus.CREATED);
    }

    @PostMapping("/login")
    public ResponseEntity<?> loginAuthentication(@Valid @RequestBody LoginRequestDTO loginRequestDTO) {
        User userDB = null;

        try{
            userDB = customUserDetailsService.getUserByUsername(loginRequestDTO.getUsername());
        } catch (Exception e) {
            throw new RequestDeniedException("Korisnik se nije mogao dohvatiti iz baze podataka.");
        }

        if (userDB == null){
            throw new RequestDeniedException("Korisničko ime ne postoji.");
        }

        if (!userDB.isAccountActivatedByEmail()){
            throw new RequestDeniedException("Molimo Vas prvo potvrdite email adresu.");
        }

        Authentication authentication = authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(
                        loginRequestDTO.getUsername(),
                        loginRequestDTO.getPassword()
                )
        );

        SecurityContextHolder.getContext().setAuthentication(authentication);

        String jwt = tokenProvider.generateToken(authentication);
        Long id = tokenProvider.getUserIdFromJWT(jwt);
        User user= customUserDetailsService.getUserById(id).orElseThrow(
                () -> new EntityMissingException(User.class, id));

        return ResponseEntity.ok(new JwtAuthenticationResponse(jwt,user.getId(),user.getRole(),user.getUsername()));
    }
}