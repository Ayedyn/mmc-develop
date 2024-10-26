#pragma once

namespace mcx {
	struct Medium {
	    // The absorption coefficient, mua, in 1/mm.
		float absorption;
		// The inverse of the absorption coefficient, or 1/mua, in mm.
		float inverseAbsorption;
		// The scattering coefficient, mus, in 1/mm.
		float scattering;
		// The inverse of the scattering coefficient, or 1/mus, in mm.
		float inverseScattering;
		// The anisotropy factor, g.
		float anisotropy;
		// The inverse of one minus the anisotropy, or 1.0 / (1.0 - g).
		float inverseOneAnisotropy;
		// The refractive index, or n.
		float refractiveIndex;

		Medium() = default;

		constexpr Medium(float mua, float mus, float g, float n) : absorption(mua), inverseAbsorption(1.0 / mua), scattering(mus), inverseScattering(1.0 / mus), anisotropy(g), inverseOneAnisotropy(1.0 / (1.0 - g)), refractiveIndex(n) {
		}
	};
}