package hr.fer.progi.simplicity.security.jwt;

import hr.fer.progi.simplicity.entities.RoleType;

public class JwtAuthenticationResponse {

    private String accessToken;
    private Long id;
    private RoleType role;
    private String username;

    public JwtAuthenticationResponse(String accessToken) {
        this.accessToken = accessToken;
    }

    public JwtAuthenticationResponse(String accessToken, Long id, RoleType role, String username) {
        this.accessToken = accessToken;
        this.id = id;
        this.role = role;
        this.username = username;
    }

    public String getAccessToken() {
        return accessToken;
    }

    public void setAccessToken(String accessToken) {
        this.accessToken = accessToken;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public RoleType getRole() {
        return role;
    }

    public void setRole(RoleType role) {
        this.role = role;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }
}
