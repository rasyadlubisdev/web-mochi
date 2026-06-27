export function MochiLogo({ size = 36 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      {/* body shadow */}
      <ellipse cx="50" cy="82" rx="32" ry="8" fill="rgba(0,0,0,0.18)" />
      {/* body */}
      <ellipse cx="50" cy="56" rx="40" ry="36" fill="#f9e4f0" />
      {/* top dome highlight */}
      <ellipse cx="50" cy="38" rx="28" ry="20" fill="#fdf0f8" />
      {/* cheek blush left */}
      <ellipse cx="31" cy="60" rx="9" ry="6" fill="#f7b8d8" opacity="0.7" />
      {/* cheek blush right */}
      <ellipse cx="69" cy="60" rx="9" ry="6" fill="#f7b8d8" opacity="0.7" />
      {/* left eye */}
      <ellipse cx="39" cy="52" rx="4" ry="4.5" fill="#2d1b2e" />
      <circle cx="40.5" cy="50.5" r="1.2" fill="white" />
      {/* right eye */}
      <ellipse cx="61" cy="52" rx="4" ry="4.5" fill="#2d1b2e" />
      <circle cx="62.5" cy="50.5" r="1.2" fill="white" />
      {/* smile */}
      <path
        d="M42 62 Q50 69 58 62"
        stroke="#c0679a"
        strokeWidth="2.5"
        strokeLinecap="round"
        fill="none"
      />
      {/* tiny highlight arc on body */}
      <path
        d="M28 46 Q50 28 72 46"
        stroke="white"
        strokeWidth="3"
        strokeLinecap="round"
        opacity="0.45"
        fill="none"
      />
    </svg>
  );
}
